import os
import re
import numpy as np

import nltk.stem.porter as pt
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AdamW
from transformers import BertTokenizer, BertForNextSentencePrediction
from util.parsehtml import HtmlInfo
from evalfuntion.retrivalEval import NDCG
from sklearn.model_selection import train_test_split
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from torch.utils.tensorboard import SummaryWriter

url_re = re.compile("X-INKT-URI:(.*?)\n")

class BertForIRdataset():
    def __init__(self,idlist,inputid,masks,typeids,y_label,max_len = 512):
        super(BertForIRdataset, self).__init__()
        self.idlist = idlist
        self.inputid = inputid
        self.masks = masks
        self.typeids = typeids
        self.y_label = y_label
        self.max_len = 512
    
    def __len__(self):
        return len(self.idlist)

    def __getitem__(self, index):
        qid = self.idlist[index][0]
        docid = self.idlist[index][1]
        ids = self.inputid[index]
        mask = self.masks[index]
        token_type_ids = self.typeids[index]
        label = self.y_label[index]

        token_len = len(ids)
        if(token_len<self.max_len):
            ids = ids + [0]*(self.max_len-token_len)
            mask = mask + [0]*(self.max_len-token_len)
            token_type_ids = token_type_ids + [0]*(self.max_len-token_len)
        elif(token_len>self.max_len):
            ids = ids[:self.max_len-1] + [102]
            mask = mask[:self.max_len]
            token_type_ids = token_type_ids[:self.max_len]
            
        ids_tensor = torch.tensor(ids).cuda()
        mask_tensor = torch.tensor(mask).cuda()
        typeids_tensor = torch.tensor(token_type_ids).cuda()
        label_tensor = torch.tensor(label).cuda()
        return qid,docid,ids_tensor,mask_tensor,typeids_tensor,label_tensor

class BertForIRtokenizer():
    def __init__(self,
                 queryfile="",
                 data_path = "",
                 tag_name = "body",
                 bert_path = 'bert-base-uncased',
                 max_seqlenth=512
                ):
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.idf={}
        self.tag = tag_name
        self.max_lenth = max_seqlenth
        self.load_queryfile(queryfile)
        self.cache_dict = {}
    
    def load_queryfile(self,queryfile):
        self.query_dict={}
        f = open(queryfile, "rt", encoding="utf-8")
        xml_text = f.read()
        f.close()
        soup_string = BeautifulSoup(xml_text, "html.parser")
        query_list = soup_string.select("query")
        for query in query_list:
            query_id = query.select("qid")[0].get_text()
            content = query.select("content")[0].get_text()
            self.query_dict[query_id] = content
        return self
    
    def get_doctext(self,docid):
        file=self.data_path+"/" + docid+".html"
        if(not os.path.exists(file)):
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

    def getTokenlist(self,qid,docid):
        query_text = self.query_dict[qid]
        if(self.cache_dict.get(qid,{}).get(docid,[])):
            return self.cache_dict[qid][docid][0],self.cache_dict[qid][docid][1],self.cache_dict[qid][docid][2]
        doc_text = self.get_doctext(docid)
        encoding = self.tokenizer(query_text, doc_text, max_length = self.max_lenth ,padding='max_length',)
        self.cache_dict[qid] = self.cache_dict.get(qid,{})
        self.cache_dict[qid][docid] = [encoding["input_ids"],encoding["attention_mask"],encoding["token_type_ids"]]
        return self.cache_dict[qid][docid][0],self.cache_dict[qid][docid][1],self.cache_dict[qid][docid][2]
    
    def cache_clear(self):
        self.cache_dict = {}

class BERTforIR():
    def __init__(self,
                 max_seqlenth=512,
                 learning_rate=1e-5,
                 train_flag=True,
                 batch_size = 8,
                 bert_path = 'bert-base-uncased',
                 model_path = "/home/user/ntcir_match/lzdeep/modelpath/BERT/"):
        self.max_seqlenth=max_seqlenth
        self.batch_size = batch_size
        self.model_path = model_path
        self.evalmodel = NDCG()
        if(train_flag):
            self.model = BertForNextSentencePrediction.from_pretrained(bert_path).cuda()
            self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        else:
            self.model = self.load_model(model_path)
        pass

    def train(self,data,y_label,qid,BertLoadear,epoch=1000,max_step=40):
        train_id,val_id = train_test_split(qid,test_size=0.1)
        traindata = []
        valdata = []
        y_train = []
        y_val = []
        for batch,label in zip(data,y_label):
            qid,docid = batch
            if(qid in train_id):
                if(label>1):
                    label = 1
                traindata.append([qid,docid])
                y_train.append(label)
            elif(qid in val_id):
                valdata.append([qid,docid])
                y_val.append(label)
        print("Data has been splited.....")
        train_inputid = []
        train_mask = []
        train_typeids = []
        tag_num = 0
        for qid,docid in traindata:
            input,mask,typeids = BertLoadear.getTokenlist(qid,docid)
            train_inputid.append(input)
            train_mask.append(mask)
            train_typeids.append(typeids)
            tag_num = tag_num + 1 
            if(tag_num%1000==0):
                print(str(tag_num) + " has been loaded....")
        train_data = BertForIRdataset(traindata,train_inputid,train_mask,train_typeids,y_train,max_len=self.max_seqlenth) 
        train_loader = DataLoader(train_data,batch_size=self.batch_size)
        stop_batch = 0
        pre_value = 0.0
        for num in range(int(epoch)):
            self.model.train()
            print("- for epoch_num:{} in max_epoch:{}".format( num , epoch ))
            tqdm_bar = tqdm(train_loader, desc="Training")
            running_loss = 0.0
            step_num = 0
            for step, batch in enumerate(tqdm_bar):
                qid,docid,inputids,masks,type_ids,label = batch
                output = self.model(input_ids=inputids, attention_mask=masks, token_type_ids=type_ids,labels=label)
                
                loss = output[0]
                y_ = output[1]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                step_loss = loss.item()
                running_loss +=step_loss
                #tqdm_bar.set_postfix(loss=step_loss,acc=acc_tmp/len(batch_y.tolist()), str="h", step=step)
                tqdm_bar.set_postfix(loss=step_loss, str="h", step=step)
                step_num = step_num+1
            self.model.eval()
            all_loss = running_loss / step_num

            val_posdata = []
            y_label = []
            sim_dict = {}
            for idbatch,label in zip(valdata,y_val):
                qid,pos_id = idbatch
                sim_dict[qid] = sim_dict.get(qid,{})
                sim_dict[qid][pos_id] = label
                val_posdata.append([qid,pos_id])
            result_dict = self.predict(val_posdata,BertLoadear,test_batchsize=self.batch_size)
            ndcg_value = []
            for key in sim_dict.keys():
                eval_result = []
                for docid in sim_dict[key].keys():
                    eval_result.append([result_dict[key][docid],sim_dict[key][docid]])
                self.evalmodel.covertList(eval_result,simindex=0,relindex=1)
                ndcg_value.append(self.evalmodel.getDcg(10))
            mean_ndcg = sum(ndcg_value)/len(ndcg_value)

            stop_batch = stop_batch + 1
            #print(test_y)
            print("- train_data loss /{:04.2f}/ - dev_data ndcg@10 /{:04.2f}/".format(all_loss , mean_ndcg))
            tqdm_bar.close()
            if (mean_ndcg > pre_value):
                pre_value = mean_ndcg
                print("Get new best score!")
                self.save_model(self.model_path)
                stop_batch = 0
            if (stop_batch > max_step):
                break
            tqdm_bar.close()

    def save_model(self, modelpath):
        if not os.path.exists(modelpath):
            os.makedirs(modelpath)
        model_name = "model.pkl"
        torch.save(self.model, modelpath+model_name)
        return self
    
    def load_model(self, modelpath):
        if not os.path.exists(modelpath):
            os.makedirs(modelpath)
        model_name = "model.pkl"
        self.model = torch.load(modelpath+model_name)
        return self.model

    def predict(self,data,BertLoadear,test_batchsize = 8):
        result_dict = {}
        test_inputid = []
        test_mask = []
        test_typeids = []
        for qid,docid in data:
            input,mask,typeids = BertLoadear.getTokenlist(qid,docid)
            test_inputid.append(input)
            test_mask.append(mask)
            test_typeids.append(typeids)
        label_ = [0]*len(data)
        self.model.eval()
        with torch.no_grad():
            test_data = BertForIRdataset(data,test_inputid,test_mask,test_typeids,label_,max_len=self.max_seqlenth) 
            test_loader = DataLoader(test_data,batch_size=test_batchsize)
            for batch in test_loader:
                qid,docid,inputids,masks,type_ids,_ = batch
                rel_value = self.model(input_ids=inputids, attention_mask=masks, token_type_ids=type_ids)[0]
                rel_value = torch.index_select(rel_value,1,torch.tensor([1]).cuda()).view(-1)
                for qid,docid,cosine in zip(qid,docid,rel_value.tolist()):
                    result_dict[qid] = result_dict.get(qid,{})
                    result_dict[qid][docid] = cosine
        return result_dict