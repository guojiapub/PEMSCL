python ../codes/train.py --data_dir ../dataset/re-docred/ \
                --transformer_type roberta  \
                --model_name_or_path roberta-large \
                --train_file train_annotated.json  \
                --dev_file dev.json \
                --test_file test.json \
                --train_batch_size 4 \
                --test_batch_size 4 \
                --gradient_accumulation_steps 1 \
                --num_labels 4 \
                --learning_rate 2e-5 \
                --max_grad_norm 1.0 \
                --warmup_ratio 0.06 \
                --num_train_epochs 8.0 \
                --seed 66 \
                --num_class 97 \
                --gpu 1 \
                --tau 0.2 \
                --tau_base 2.0 \
                --save_path ../checkpoints/re-docred.pt