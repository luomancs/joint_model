import json, os
import random
from tqdm import tqdm
import numpy as np
import string 
import math


from common.utils_plus import *

# this function process the raw hotpotQA data to the desired input format. one example of the input format is given below. 
'''
{"para": "The arena where the Lewiston Maineiacs played their home games can seat how many people?</s> The Androscoggin Bank Colisée (formerly Centra
l Maine Civic Center and Lewiston Colisee) is a 4,000 capacity (3,677 seated) multi-purpose arena, in Lewiston, Maine, that opened in 1958.</s>  In 1
965 it was the location of the World Heavyweight Title fight during which one of the most famous sports photographs of the century was taken of Muham
med Ali standing over Sonny Liston.", "para_label": 1, "sents_label": [1, -1]}
'''

def hotpot(read_path, save_path,mode="train"):
    posi_para = []
    neg_para = []
    data = json_plus.load(read_path)
    for item in tqdm(data):
        question = item['question'] 
        answer = item['answer']
        
        supporting_title = {}
        for sp in item['supporting_facts']:
            title = sp[0]
            idx = sp[1]
            if title not in supporting_title:
                supporting_title[title] = [idx]
            else:
                supporting_title[title].append(idx)

        for para in item['context']:
            title = para[0]
            sents_label = [-1]*(len(para[1]))
            para_label = -1
            if  title in supporting_title:
                para_label = 1
                for idx in supporting_title[title]:
                    if  idx < len(para[1]):
                        sents_label[idx] = 1
            
            datum = {
                'para':"</s> ".join([question]+para[1][:9]),
                'para_label':para_label,
                'sents_label':sents_label[:9],
                'target_text': answer,
            }
            if title in supporting_title:
                posi_para.append(datum)
            else:
                neg_para.append(datum)
    
    if mode=="train":
        neg_para = random.sample(neg_para, len(posi_para))
    all_samples = posi_para+neg_para
    print(len(all_samples))
    jsonline_plus.dump(all_samples, save_path)

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))

def normalize(str1):
    str2 = str1.translate(translator)
    str2 = str2.lower().replace(" ","")
    return str2

path = '../dataset/para_training/'
hotpot(path+'hotpot_train_v1.1.json', path+'train.jsonl', mode='train')