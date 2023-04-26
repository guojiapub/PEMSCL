import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class PMTEMloss(nn.Module):
    def __init__(self, lambda_1, lambda_2):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2


    def forward(self, logits, labels):
        label_mask = (labels[:, 0] != 1.)

        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

 
        count = labels.sum(1, keepdim=True)
        count[count==0] = 1


        th = logits[:, :1].expand(logits.size(0), logits.size(1))

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        probs1 = F.softmax(torch.cat([th.unsqueeze(1), logit1.unsqueeze(1)], dim=1), dim=1)
        loss1 = -(torch.log(probs1[:, 1] + 1e-30) * labels).sum(1)
        loss3 = -(((probs1 * torch.log(probs1 + 1e-30)).sum(1)) * labels ).sum(1) / count


        count = (1-p_mask).sum(1, keepdim=True)
        count[count==0] = 1


        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        probs2 = F.softmax(torch.cat([th.unsqueeze(1), logit2.unsqueeze(1)], dim=1), dim=1)
        loss2 = -(torch.log(probs2[:, 0] + 1e-30) * (1 - p_mask)).sum(1)
        loss4 = -(((probs2 * torch.log(probs2 + 1e-30)).sum(1)) * (1 - p_mask)).sum(1) / count

            
        # Sum  parts
        loss = loss1 + loss2 + self.lambda_1*loss3 + self.lambda_2*loss4
        loss = loss.mean()
        return loss


    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output




class MLLTRSCLloss(nn.Module):
    def __init__(self, tau=2.0, tau_base=1.0):
        super().__init__()
        self.tau = tau
        self.tau_base = tau_base

    def forward(self, features, labels, weights=None):
        labels = labels.long()
        label_mask = (labels[:, 0] != 1.)
        mask_s = torch.any((labels.unsqueeze(1) & labels).bool(), dim=-1).float().fill_diagonal_(0)

        sims = torch.div(features.mm(features.T), self.tau)
        
        logits_max, _ = torch.max(sims, dim=1, keepdim=True)
        logits = sims - logits_max.detach()
        logits_mask = torch.ones_like(mask_s).fill_diagonal_(0)

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        denom = mask_s.sum(1)
        denom[denom==0] = 1     # avoid div by zero

        log_prob1 = (mask_s * log_prob).sum(1) / denom

        log_prob2 = - torch.log(exp_logits.sum(-1, keepdim=True)) * (mask_s.sum(-1, keepdim=True) == 0)
        mean_log_prob_pos = (log_prob1 + log_prob2) * label_mask

        loss = - (self.tau/self.tau_base) * mean_log_prob_pos
        loss = loss.mean()

        return loss




class PMTEMloss_NS(nn.Module):
    def __init__(self, lambda_1, lambda_2, ratio):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.ratio = ratio
        
    def forward(self, logits, labels):
        label_mask = (labels[:, 0] != 1.)

        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        th = logits[:, :1].expand(logits.size(0), logits.size(1))

        count = labels.sum(1, keepdim=True)
        count[count==0] = 1

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        probs1 = F.softmax(torch.cat([th.unsqueeze(1), logit1.unsqueeze(1)], dim=1), dim=1)
        loss1 = -(torch.log(probs1[:, 1] + 1e-30) * labels).sum(1)
        loss3 = -(((probs1 * torch.log(probs1 + 1e-30)).sum(1)) * labels ).sum(1) / count


        random_mask = torch.tensor(np.random.uniform(0,1,97) > self.ratio).to(n_mask)
        random_mask[0] = False

        count = (1-p_mask).sum(1, keepdim=True)
        count[count==0] = 1

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30 - (random_mask.unsqueeze(0) * ~label_mask.unsqueeze(1))*1e30
        probs2 = F.softmax(torch.cat([th.unsqueeze(1), logit2.unsqueeze(1)], dim=1), dim=1)
        loss2 = -(torch.log(probs2[:, 0] + 1e-30) * (1 - p_mask)).sum(1)
        loss4 = -(((probs2 * torch.log(probs2 + 1e-30)).sum(1)) * (1 - p_mask)).sum(1) / count

        # Sum  parts
        loss = loss1 + loss2 + self.lambda_1*loss3 + self.lambda_2*loss4
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output





