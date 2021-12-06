# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

import logging
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional, Union

from transformers.file_utils import is_tf_available
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.data.processors.utils import DataProcessor, InputExample

import jsonlines

from tqdm import tqdm
import random
import json
import pickle
import csv

import numpy as np

class json_plus():
    @staticmethod
    def load(infile):
        with open(infile, 'r') as df:
            f = json.load(df)
        return f

    @staticmethod
    def dump(payload, outfile):
        with open(outfile, 'w') as df:
            f = json.dump(payload, df)
        return f


class pickle_plus():
    @staticmethod
    def load(infile):
        with open(infile, 'rb') as df:
            f = pickle.load(df)
        return f

    @staticmethod
    def dump(payload, outfile):
        with open(outfile, 'wb') as df:
            f = pickle.dump(payload, df)
        return f

class csv_plus():
    @staticmethod
    def dump(content, write_path):
        csv_columns = list(content[0].keys())
        with open(write_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in content:
                writer.writerow(data)
    
    @staticmethod
    def load(infile):
        return_row = []
        with open(infile, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                return_row.append(row)
        return return_row


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def glue_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        if task is None:
            raise ValueError("When calling glue_convert_examples_to_features from TF, the task parameter is required.")
        return _tf_glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
    

    return _glue_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )


if is_tf_available():

    def _tf_glue_convert_examples_to_features(
        examples: tf.data.Dataset, tokenizer: PreTrainedTokenizer, task=str, max_length: Optional[int] = None,
    ) -> tf.data.Dataset:
        """
        Returns:
            A ``tf.data.Dataset`` containing the task-specific features.

        """
        processor = glue_processors[task]()
        examples = [processor.tfds_map(processor.get_example_from_tensor_dict(example)) for example in examples]
        features = glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)

        def gen():
            for ex in features:
                d = {k: v for k, v in asdict(ex).items() if v is not None}
                label = d.pop("label")
                yield (d, label)

        input_names = ["input_ids"] + tokenizer.model_input_names

        return tf.data.Dataset.from_generator(
            gen,
            ({k: tf.int32 for k in input_names}, tf.int64),
            ({k: tf.TensorShape([None]) for k in input_names}, tf.TensorShape([])),
        )

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self,
               input_ids,
               attention_mask,
               label= None,
               start_positions=None,
               end_positions=None):
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.start_positions = start_positions
    self.end_positions = end_positions
    self.label= label

def _glue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
           
            return label_map[example.label]
        elif output_mode == "regression":
            if isinstance(example.label, list):
                return [float(i) for i in example.label] 
            return float(example.label)
        raise KeyError(output_mode)
    labels = [label_from_example(example) for example in examples]
    #examples=examples[:10000]
    #labels= labels[:10000]
    features = []
    for i,example in enumerate(examples):
        x = {}
        x['para']=example.text_a
        x['target_text'] = example.text_b
        tokenized_example = prepare_train_features(x,tokenizer)
        y={}
        if len(tokenized_example['input_ids']) ==1:
          y['input_ids'] = tokenized_example['input_ids'][0]
          y['attention_mask'] = tokenized_example['attention_mask'][0]
          y['start_positions'] = tokenized_example['start_positions']
          y['end_positions'] = tokenized_example['end_positions']

          feature = InputFeatures(**y, label=labels[i])
          features.append(feature)
    print(len(features))

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features

def prepare_train_features(examples,tokenizer,pad_on_right=True,max_length=512,doc_stride=128):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    
    tokenized_examples = tokenizer(
            examples["para"],
            #truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
    

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    remove = []
    # print(tokenized_examples)
    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        
        input_ids = tokenized_examples["input_ids"][i]
        
        cls_index = input_ids.index(tokenizer.eos_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
#         answer = examples["answers"][sample_index]
        answer = examples['target_text']
        # If no answers are given, set the cls_index as answer.
#         try:
        # Start/end character index of the answer in the text.
        context = examples['para']
        
        
        start_char = context.lower().find(answer.lower())
        if start_char == -1: # not find
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            remove.append(i)
            continue
            
        
        end_char = start_char + len(answer)

        # Start token index of the current span in the text.
        token_start_index = 0
        
        while sequence_ids[token_start_index] != 0:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] !=0:
            token_end_index -= 1
        
        # Detect if the answer is out of the span (in which case we remove this split).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            remove.append(i)
        
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
    new_input_ids = []
    new_target_ids = []
    for i,(ids, attn) in enumerate(zip(tokenized_examples['input_ids'],tokenized_examples['attention_mask'])):
        if i not in remove:
            new_input_ids.append(ids)
            new_target_ids.append(attn)
    tokenized_examples['input_ids'] = new_input_ids
    tokenized_examples['attention_mask']=new_target_ids
    return tokenized_examples


class OutputMode(Enum):
    classification = "classification"
    regression = "regression"



class PARAProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def _read_jsonl(self,fname):
        with jsonlines.open(fname,"r") as f:
            return [ x for x in f]
        
    def get_train_examples(self, data_dir):
        """See base class."""
        data1 = []
        data2 = []
        data1 = self._read_jsonl(os.path.join(data_dir, "train.jsonl"))
        # data2 = self._read_jsonl(os.path.join(data_dir, "train_beth.jsonl"))
        return self._create_examples(data1+data2, "train",data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev",data_dir)

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "test",data_dir)

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type,data_dir):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, row) in tqdm(enumerate(lines),desc="Generating"):
            guid = "%s-%s" % (set_type, i)
            text_a = row['para']
            text_b = row['target_text']
            label = [row['para_label']]+ row['sents_label'] 
            label = label + [0]*(10-len(label)) 
            
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


glue_tasks_num_labels = {

    "para":1,

    
}

glue_processors = {
    
    "para":PARAProcessor,

   
}

glue_output_modes = {
    
    "para":"regression",

}

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        
        return {"acc":(preds == labels).mean()}

    def regression_accuracy(preds, labels):
        preds = preds >0
        labels = labels >0
        res = acc_and_f1(preds, labels)
        return res
        # return (preds == labels).mean()
    def para_accuracy(preds, labels):
        para_pred = preds 
        para_pred = para_pred >0
        para_labels = labels[:,0] >0
        para_res = acc_and_f1(para_pred, para_labels)

        
        
        res = {
            'para_acc':para_res['acc'],
            'para_f1':para_res['f1'], 
            # 'sent_acc':sents_res['acc'],
            # 'sent_f1':sents_res['f1']
        }
        return res
        

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
        return {
            "acc": acc,
            "f1": f1,
            
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def glue_compute_metrics(task_name, preds, labels):
        
        # assert len(preds) == len(labels)
        
        if task_name == "para":
            return {"acc": para_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)
        
    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)