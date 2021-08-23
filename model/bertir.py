import argparse
import glob
import logging
import os
import random
import timeit

import sys

sys.path.append(r'/home/user/ntcir_match/lzdeep/NTCIR_project/')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
import re
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForNextSentencePrediction
from util.parsehtml import HtmlInfo
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from torch.utils.tensorboard import SummaryWriter

class BERT_IR:
    def __init__(self,
                 data_path="/home/user/ntcir_match/en_task/result_html_eng/",
                 model_path="/home/user/ntcir_match/lzdeep/modelpath/drmm/BERT",
                 max_seqlenth=512,
                 weight_decay=0.0,
                 grad_norm=1.0,
                 num_train_epochs=2000,
                 max_steps=10,
                 seed=42,
                 queryfile=""):
        self.data_path=data_path
        self.modelpath=model_path
        self.max_seqlenth=max_seqlenth
        self.weight_decay=weight_decay
        self.grad_norm=grad_norm
        self.num_train_epochs=num_train_epochs
        self.max_steps=max_steps
        self.seed=seed
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        self.querydict=self.load_queryfile(queryfile)
        self.evaldict={}
        self.value_re=self.value_re=re.compile("[a-zA-Z]")
        self.testdata={}
        self.train_data=[]
        self.train_y=[]
        self.val_data=[]
        self.val_y=[]
        self.optimizer = AdamW(self.model.named_parameters(), lr=0.01)

    def load_queryfile(self, queryfile):
        query_dict = {}
        f = open(queryfile, "rt", encoding="utf-8")
        xml_text = f.read()
        f.close()
        soup_string = BeautifulSoup(xml_text, "html.parser")
        query_list = soup_string.select("query")
        for query in query_list:
            query_id = query.select("qid")[0].get_text()
            content = query.select("content")[0].get_text()
            # self.querydict[query_id] = content.split(" ")
            query_dict[query_id] = content
        return query_dict

    def load_baseline(self, basefile, qIndex=0, subqindex=-1, dIndex=2, split_tag=" "):
        f = open(basefile, "rt", encoding="utf-8")
        for line in f.readlines():
            datas = line.split(split_tag)
            if (len(datas) < max([qIndex, subqindex, dIndex])):
                continue
            q_id = datas[qIndex]
            if (subqindex != -1):
                q_id = q_id + "_" + datas[subqindex]
            doc_id = datas[dIndex]
            self.testdata[q_id]=doc_id
        f.close()
        return self

    def load_evalfile(self, similarFile, qIndex=0, subqindex=-1, dIndex=1, simIndex=2, split_tag=" "):
        """
        :param similarFile:人工相似度标签
        :param qIndex: 相似度文档的 query 索引
        :param subqindex:相似度文档的 subtopic 索引，不存在则为-1
        :param dIndex:相似度文档的 docid 索引
        :param simIndex:相似度文档的 相似度 索引
        :param split_tag:分隔符
        :return:self
        """
        f = open(similarFile, "rt", encoding="utf-8")
        maxindex = max([qIndex, dIndex, simIndex])
        line = f.readline()
        while (line):
            datas = line.replace("\r", "").replace("\n", "").split(split_tag)
            if (len(datas) < maxindex):
                line = f.readline()
                continue
            if (subqindex != -1):
                q_id = datas[qIndex] + "_" + datas[subqindex]
            else:
                q_id = datas[qIndex]
            doc_id = datas[dIndex]
            sim = self.get_value(datas[simIndex])
            self.evaldict[q_id] = self.evaldict.get(q_id, {})
            self.evaldict[q_id][doc_id] = int(sim)
            line = f.readline()
        f.close()
        return self

    def get_value(self,value):
        return self.value_re.sub("",value)

    def shuffle_data(self,ratio=0.9):
        for q_id in self.evaldict.keys():
            for doc_id in self.evaldict[q_id].keys():
                all_text=self.get_text(doc_id)
                query_text=self.querydict[q_id]
                tokenized_sequence = self.tokenizer.encode_plus(query_text, all_text,max_length = self.tokenizer.model_max_length,pad_to_max_length=True)
                inputs=self.norm_input(tokenized_sequence)
                value=random.random()
                y_label=[0,0]
                if(self.evaldict[q_id][doc_id]):
                    y_label[0]=1
                else:
                    y_label[1]=1
                if(value<ratio):
                    self.train_data.append(inputs)
                    self.train_y.append(y_label)
                else:
                    self.val_data.append(inputs)
                    self.val_y.append(y_label)
        return self

    def norm_input(self,seqdict):
        tmp_dict={}
        tmp_dict["input_ids"]=torch.tensor(seqdict["input_ids"],dtype=torch.long).unsqueeze(0)
        tmp_dict["attention_mask"]=torch.tensor(seqdict["attention_mask"],dtype=torch.float32).unsqueeze(0)
        tmp_dict["token_type_ids"]=torch.tensor(seqdict["token_type_ids"],dtype=torch.long).unsqueeze(0)
        return tmp_dict

    def get_text(self,docid):
        file = self.data_path + "/" + docid + ".html"
        if (not os.path.exists(file)):
            return []
        f = open(file, "rb")
        data = f.read().decode("utf-8", "ignore")
        f.close()
        html_code = "\n".join(data.split("\r\n\r\n")[1:])

        maininfo = HtmlInfo(html_code)
        a_text = maininfo.get_text("a")
        title_text = maininfo.get_text("title")
        body_text=maininfo.get_text("body")
        html_text=maininfo.get_text("html")
        all_text=" ".join([a_text,title_text,body_text,html_text])
        return all_text

    def train(self):
        pre_loss=None
        stop_batch=0
        for _ in range(self.num_train_epochs):
            running_loss=0.0
            for i in range(len(self.train_data)):
                inputs=self.train_data[i]
                y_label=self.train_y[i]
                inputs["next_sentence_label"]=y_label
                self.model.zero_grad()
                outputs = self.model(**inputs)
                loss=outputs[0]
                loss.backward()
                self.optimizer.step()
                running_loss+=loss.item()
                if(i%1000):
                    print('[%d, %5d] loss: %.3f' %(_, i+1, running_loss/1000))
                    running_loss=0.0
            if (_ % 5 == 0 and _ != 0):
                val_loss = 0.0
                num = len(self.val_data)
                for i in range(len(self.train_data)):
                    inputs = self.train_data[i]
                    y_label = self.train_y[i]
                    inputs["next_sentence_label"] = y_label
                    outputs = self.model(**inputs)
                    loss = outputs[0].mean()
                    val_loss+=loss.item()
                val_loss = val_loss / num
                print("第" + str(_) + "次实验的loss:" + str(val_loss))
                stop_batch = stop_batch + 1
                if (not pre_loss):
                    pre_loss = val_loss
                if (pre_loss > val_loss and pre_loss):
                    pre_loss = val_loss
                    self.model.save_pretrained(self.modelpath)
                    stop_batch = 0
                if (stop_batch > self.max_steps):
                    break
        return self

    def eval_test(self):
        result={}
        for qid in self.testdata.keys():
            result[qid]={}
            for docid in self.testdata[qid].keys():
                all_text = self.get_text(docid)
                query_text = self.querydict[qid]
                tokenized_sequence = self.tokenizer.encode_plus(query_text, all_text)
                inputs = self.norm_input(tokenized_sequence)
                outpus=self.model(**inputs)
                rankvalue=outpus[0][0][0].item()
                result[qid][docid]=rankvalue
        return result

if __name__=="__main__":
    bertir=BERT_IR(data_path="/home/user/ntcir_match/en_task/result_html_eng/",
                 model_path="/home/user/ntcir_match/lzdeep/modelpath/drmm/BERT",
                 max_seqlenth=512,
                 weight_decay=0.0,
                 grad_norm=1.0,
                 num_train_epochs=2000,
                 max_steps=10,
                 seed=42,
                 queryfile="")
