PEMSCL
===
This is the official implementation for "[Towards Integration of Discriminability and Robustness for Document-Level Relation Extraction](https://arxiv.org/abs/2304.00824)" (EACL 2023, Main Conference).

## Citation
If you find our work helpful for your research, please cite our paper:

```bibtex
@inproceedings{pemscl,
    title = "Towards Integration of Discriminability and Robustness for Document-Level Relation Extraction",
    author = "Jia Guo and Stanley Kok and Lidong Bing",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    year = "2023"
}
```

## Installation
```bash
>> conda create -n pemscl python==3.7.4
>> conda activate pemscl
>> conda install pytorch==1.7.0 torchvision cudatoolkit=10.2 -c pytorch
```
Other dependencies:
- tqdm==4.61.1
- transformers==4.8.2
- ujson==4.0.2
- numpy==1.19.4
- apex==0.1
- opt-einsum==3.3.0

The expected folder structure is:

```
PEMSCL
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json        
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |-- re-docred
 |    |    |-- train_annotated.json        
 |    |    |-- dev.json
 |    |    |-- test.json
 |-- meta
 |    |-- rel2id.json
 |-- codes
 |-- scripts 
 |-- results
 |-- logs
 |-- checkpoints
```

## Training models on benchmarks
```bash
>> cd scripts
>> sh run_docred.sh  # for the DocRED dataset
>> sh run_re-docred.sh # for the Re-DocRED dataset
```


## Evaluation
You can evaluate your saved checkpoint by replacing the `--save_path` argument with the `--load_path` argument. Our trained models `pemscl-docred` and `pemscl-re-docred` can be found [here](https://drive.google.com/drive/folders/1iaHDreWu0xtBnzjOTNRzNyp2OnemvNmA?usp=share_link).


## Acknowledgement
We refer to the codes of [ATLOP](https://github.com/wzhouad/ATLOP). Thanks for their contributions.

