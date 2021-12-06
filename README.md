# A Simple Approach to Jointly Rank Passages and Select Relevant Sentences in the OBQA Context

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository contains the implementations of the system described in the paper ["A Simple Approach to Jointly Rank Passages and Select Relevant Sentences in the OBQA Context"](https://arxiv.org/pdf/2109.10497.pdf)

The motivation of this work is to build a joint model For QA such that it can do both passage ranking and relevant sentence classification. Compared to systems which rely on multiple models to do each task, our joint model has less parameters while achieving reasonable performance. Please refer to the [paper](https://arxiv.org/pdf/2109.10497.pdf) for details.

## Installation
The code work with Python 3.7. If you use conda, you can set up the environment as follows:
```bash
conda create -n env_name python==3.7
conda activate env_name
```

Also, install the dependencies specified in the requirements.txt:
```
pip install -r requirements.txt
```

## Data
You can download the preprocess data of HotpotQA with the following links: [data](https://drive.google.com/drive/folders/1rUth38_A6kgOqzCJeonIAZ7tIJPv9HRr?usp=sharing). The prepocessed data has concatenated the question and passage as input and injected special token `</s>` before each sentence in a passage. 
Download the data into `dataset/` folder. We provide two toy examples under the `dataset/toy.jsonl`.

#### Data pre-processing
We provide a function under `process_data/process_hotpotqa.py`to preprocess the [raw HotpotQA data](https://hotpotqa.github.io/). You can use it to convert the data to the jsonl format for experiments. 


## Experiment

### Training
You can train a model with the following command on your command terminal or you can use NLP_Project_Main.ipynb for the training purpose, we use RoBERTa model, you can use any other encoder-based model, like BERT and ELECTRA. 
```
python model_train/train_para.py \
--model_name_or_path roberta-base \
--tokenizer_name roberta-base \
--task_name para \
--data_dir dataset/para_training/ \
--max_seq_length 512 \
--output_dir path_to_save_model \
--do_train \
--do_eval \
--overwrite_output_dir \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--per_device_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 5 \
--save_steps 10000 \
--logging_steps 10000 \
--use_sent_loss \
--use_qa_loss \
```

## Citation
```
@article{Luo2021ASA,
  title={A Simple Approach to Jointly Rank Passages and Select Relevant Sentences in the OBQA Context},
  author={Man Luo and Shuguang Chen and Chitta Baral},
  journal={ArXiv},
  year={2021},
  volume={abs/2109.10497}
}
```

## Contact
Feel free to get in touch via email to mluo26@asu.edu.