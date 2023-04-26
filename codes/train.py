import os
import sys
import time
import copy
import torch
import pickle
import random
import logging
import argparse
import numpy as np
from apex import amp
import ujson as json
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import *
st = time.time()




def train(args, logger, config, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch):
        best_score = -1
        set_seed(args)
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False, num_workers=0, worker_init_fn=np.random.seed(args.seed))
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        logger.info("Total steps: {}".format(total_steps))
        logger.info("Warmup steps: {}".format(warmup_steps))
        num_steps = 0
        for epoch in train_iterator:
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],                    
                          }

                outputs = model(**inputs)
                loss = (outputs[0]) / args.gradient_accumulation_steps

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1


                if (step + 1) == len(train_dataloader) - 1:
                    dev_score, dev_output, dev_loss = evaluate(args, model, dev_features, tag="dev")
                    logger.info(dev_output)
                    logger.info(f'dev_loss={dev_loss.item():.4f}, step={num_steps}')

                    if dev_score > best_score:
                        best_score = dev_score
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)


    def init_model():
        set_seed(args)
        model = create_model(args, config)
        new_layer = ["extractor", "bilinear"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        return model, optimizer

    model, optimizer = init_model()
    finetune(train_features, optimizer, args.num_train_epochs)





def evaluate(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    logits = []
    topk_scores = []
    topk_index = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'labels': batch[2],
                  }

        tmp = [torch.tensor(label) for label in batch[2]]
        batch_labels = torch.cat(tmp, dim=0)

        with torch.no_grad():
            loss, pred, logit, bl = model(**inputs)
            logits.append(logit.cpu().numpy());
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
        loss = loss / args.gradient_accumulation_steps
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _, precision, recall = official_evaluate(ans, args.data_dir, mode=tag)
        output = {
            tag + "_precision": f'{precision * 100:.4f}',
            tag + "_recall": f'{recall * 100:.4f}',
            tag + "_F1": f'{best_f1 * 100:.4f}',
            tag + "_F1_ign": f'{best_f1_ign * 100:.4f}',
        }

    else:
        best_f1, _, best_f1_ign, _, precision, recall = 0.,0.,0.,0.,0.,0.
        output = {
            tag + "_precision": 'NaN',
            tag + "_recall": 'NaN',
            tag + "_F1": 'NaN',
            tag + "_F1_ign": 'NaN',
        }
    return best_f1, output, loss

def report(args, model, features):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    topk_scores = []
    topk_index = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, logit, bl = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


def create_model(args, config):
    pretrained_model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    model = DocREModel(config, args, pretrained_model, num_labels=args.num_labels)
    model.to(args.device)
    return model




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="the output dir for saving pretrained model and tokenizer")
    parser.add_argument("--tau", default=2.0, type=float, help="tau")
    parser.add_argument("--tau_base", default=1.0, type=float, help="tau_base")
    parser.add_argument("--lambda_1", default=1.0, type=float, help="lambda_1")
    parser.add_argument("--lambda_2", default=1.0, type=float, help="lambda_2")
    parser.add_argument("--lambda_3", default=1.0, type=float, help="lambda_3")
    parser.add_argument("--sample_ratio", default=1.0, type=float, help="sample_ratio")
    args = parser.parse_args()

    pt_name = args.save_path.split("/")[-1] if args.save_path != "" else args.load_path.split("/")[-1]+"_[testing]"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

    stderrLogger = logging.StreamHandler(sys.stdout)
    stderrLogger.setLevel(logging.INFO)
    stderrLogger.setFormatter(formatter)

    fileLogger = logging.FileHandler(f'../logs/{pt_name.replace(".pt", "")}.log', mode='w')
    fileLogger.setLevel(logging.INFO)
    fileLogger.setFormatter(formatter)

    logger.addHandler(stderrLogger)
    logger.addHandler(fileLogger)


    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_docred

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read(dev_file, tokenizer,  max_seq_length=args.max_seq_length)
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    set_seed(args)

    if args.load_path == "":  # Training
        train(args, logger, config, train_features, dev_features, test_features)
        logger.info(args.save_path)
    else:  # Testing
        model = create_model(args, config)
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output, _ = evaluate(args, model, dev_features, tag="dev")
        logger.info(dev_output)

        if args.data_dir == "../dataset/docred/":
            pred = report(args, model, test_features)
            with open(f'../results/result_{args.load_path.split("/")[-1][:-3]}.json', "w") as fh:
                json.dump(pred, fh)
        elif args.data_dir == "../dataset/re-docred/":
            test_score, test_output, _ = evaluate(args, model, test_features, tag="test")
            logger.info(test_output)
            pred = report(args, model, test_features)
            with open(f'../results/result_{args.load_path.split("/")[-1][:-3]}.json', "w") as fh:
                json.dump(pred, fh)


        logger.info(args.load_path)
    logger.info(args)
    logger.info(f"Time cost: {time.time()-st}")

if __name__ == "__main__":
    main()

