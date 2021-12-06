
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

from collections import defaultdict
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from transformers import RobertaTokenizerFast
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)
import torch.nn as nn

from transformers import AutoModel, AutoConfig
from transformers import PreTrainedModel, BertPreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
# from transformers import _BaseAutoModelClass
from torch.nn import MSELoss, CrossEntropyLoss, TripletMarginLoss, TripletMarginWithDistanceLoss
import torch


logger = logging.getLogger(__name__)

class ClassificationSent(nn.Module):
    """Sent for sentence-level classification tasks."""

    def __init__(self, config):
        super(ClassificationSent, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, features, input_ids, labels=None):
        x = []
        sep_count =[]
        sent_labels= []
        for i, ids in enumerate(input_ids):
            
            sep = torch.where(ids==2)
            
            if (sep[0][-1]<ids.size(0)-1 and ids[sep[0][-1]+1] == 1) or sep[0][-1]==ids.size(0)-1: # if the last <sep> is the end of para graph
                x.append(features[i,sep[0][:-1],:])
                trip_sep = sep[0][:-1].size(0)   
            else:
                x.append(features[i,sep[0],:])
                trip_sep = sep[0].size(0)
            
            sep_count.extend([i]*trip_sep)
            if labels is not None:
                sent_labels.append(labels[i,:trip_sep])
        if labels is not None:
            sent_labels = torch.cat(sent_labels, dim=0)
        sep_count = torch.tensor(sep_count)

        x = torch.cat(x, dim=0)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        y = self.out_proj(x)

        return y, x, sent_labels, sep_count


class JointQAModel(BertPreTrainedModel):
    
    # here you need to add a answer classfier to predict the start and end position
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = AutoModel.from_config(config)
        self.para_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sent_classifier = ClassificationSent(config)
        self.tokenizer=RobertaTokenizerFast.from_pretrained('roberta-base')
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.use_sent_loss = config.use_sent_loss
        self.use_qa_loss = config.use_qa_loss

        print("use sentence loss in the training: ", self.use_sent_loss)
        print("use qa loss in the training: ", self.use_qa_loss)

        self.init_weights()
        self.losses = defaultdict(list)




    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        start_positions= None, 
        end_positions = None, 
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        
        logits = self.para_classifier(outputs.pooler_output)
        
        qa_logits = self.qa_outputs(outputs.last_hidden_state)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        #print('model ansswer token = ' + str(torch.argmax(start_logits, dim=1)))
        print('model answer start string =  '+str(self.tokenizer.decode([input_ids[0][torch.argmax(start_logits, dim=1)[0]]])))
        print('model answer end string =  '+str(self.tokenizer.decode([input_ids[0][torch.argmax(end_logits, dim=1)[0]]])))

        loss = None
        output = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels[:,0].view(-1))
                self.losses['loss1'].append(loss.item())

                if self.use_sent_loss:
                    logits_sents, sents_features, sents_labels, sent_segment = self.sent_classifier(outputs.last_hidden_state, input_ids, labels[:,1:])
                    
                    loss2 = loss_fct(logits_sents.view(-1), sents_labels.view(-1))
                    loss += loss2
                    self.losses['loss2'].append(loss2.item())
                if self.use_qa_loss:
                    if start_positions is not None and end_positions is not None:
                        # If we are on multi-GPU, split add a dimension
                        if len(start_positions.size()) > 1:
                            start_positions = start_positions.squeeze(-1)
                        if len(end_positions.size()) > 1:
                            end_positions = end_positions.squeeze(-1)
                        
                        # sometimes the start/end positions are outside our model inputs, we ignore these terms
                        ignored_index = start_logits.size(1)
                        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                        start_positions = start_positions.clamp(0, ignored_index)
                        end_positions = end_positions.clamp(0, ignored_index)
                        print("Start Position: ",self.tokenizer.decode([input_ids[0][start_positions[0]]])," End Position: ", self.tokenizer.decode([input_ids[0][end_positions[0]]]))
                        start_loss = loss_fct(start_logits, start_positions)
                        end_loss = loss_fct(end_logits, end_positions)
                        loss3 = (start_loss + end_loss) / 2
                        loss += loss3
                        self.losses['loss3'].append(loss3.item())
                        #print(f'QA Loss: {loss3}')
                        self.losses['Total_Loss'].append(loss.item())
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        
        #print(f'Total Loss: {loss}')
        if not return_dict:
            output = (logits,) + outputs[2:]
            #print(outputs)
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            logits=logits,
            start_logits = start_logits,
            end_logits = end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    def predict(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_sent_loss = True,
        use_qa_loss=True
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        logits = self.para_classifier(sequence_output)
        logits_sents, sents_features, sents_labels, sent_segment = self.sent_classifier(sequence_output, input_ids, labels)
        start_positions, end_positions = self.qa_outputs(outputs.last_hidden_state)
        start_positions, end_positions = qa_logits.split(1, dim=-1)
        start_positions = start_positions.squeeze(-1).contiguous()
        end_positions = end_positions.squeeze(-1).contiguous()
        
        return {
            "para_logits":logits,
            "sent_logits" : logits_sents,
            "sent_segment":sent_segment,
            "input_f":sents_features,
            "start_logits":start_positions,
            "end_logits":end_positions,
        }
    