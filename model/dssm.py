#from data_prepro import trainData
import torch
import re
import os
import random

import sys
sys.path.append(r'/home/user/ntcir_match/lzdeep/NTCIR_project/')

alpha_re = re.compile(r"[a-z]*")

import numpy as np
import re
from bs4 import BeautifulSoup
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from util.parsehtml import HtmlInfo
from evalfuntion.retrivalEval import NDCG

import nltk.stem.porter as pt
from nltk.corpus import stopwords

pt_stemmer = pt.PorterStemmer()
pat_letter = re.compile(r'[a-zA-Z\-\'\.]+')
oth_re=re.compile(r"<!\[.*?>")
cachedStopWords = set(stopwords.words("english"))

'''
import math
class NDCG:
    def __init__(self):
        self.baseDcg=DCG()
        self.rankDcg=DCG()

    def modifybaselines_general(self,all_dict):
        """
        :param all_dict:数据规范{"docid":{"sim":人工相似度,"rank":理想排序位置}}
        :return:
        """
        self.baseDcg.modify_rank(all_dict)
        return self

    def modifyrank_general(self,all_dict):
        """
        :param all_dict:数据规范{"docid":{"sim":人工相似度,"rank":排序位置}}
        :return:
        """
        self.rankDcg.modify_rank(all_dict)
        return self

    def covertList(self,valuelist,simindex=0,relindex=1):
        """
        :param valuelist:数据规范 [[模型相似度，人工相似度]]
        :param simindex:模型相似索引
        :param relindex:人工相似度索引
        :return:
        """
        baselist=[ [value[relindex],value[relindex]] for value in valuelist]
        self.baseDcg.convertList(baselist,simindex,relindex)
        self.rankDcg.convertList(valuelist, simindex, relindex)
        return self

    def getDcg(self,n,dcgtype=""):
        """
        :param n: NDCG指标中的N值，只返回前多少结果
        :param dcgtype: DCG的计算公式分为:miDCG 和 baseDCG
        :return:
        """
        return self.rankDcg.getDcg(n,dcgtype)/self.baseDcg.getDcg(n,dcgtype)

class DCG:
    def __init__(self):
        self.data={}

    def modify_rank(self,all_dict):
        self.data={}
        for key in all_dict.keys():
            self.data[key]=all_dict[key]
        return self

    def baseDcg(self,n):
        repl=0.0
        for key in self.data.keys():
            index=self.data[key]["rank"]
            if(index>n):
                continue
            repl=repl+self.data[key]["sim"]/math.log(self.data[key]["rank"]+1,2)
        return repl
        pass

    def miDcg(self,n):
        repl = 0.0
        for key in self.data.keys():
            index = self.data[key]["rank"]
            if (index > n):
                continue
            repl = repl + (math.pow(2,self.data[key]["sim"])-1) / math.log(self.data[key]["rank"]+1, 2)
        return repl
        pass

    def convertList(self,valuelist,simindex=0,relindex=1):
        self.data={}
        finalist=sorted(valuelist,key=lambda x:x[simindex],reverse=True)
        for i in range(len(finalist)):
            self.data[i]={}
            self.data[i]["rank"]=i+1
            self.data[i]["sim"]=finalist[i][relindex]
        pass

    def getDcg(self,n,dcgtype=""):
        if(dcgtype=="miDcg"):
            return self.miDcg(n)
        else:
            return self.baseDcg(n)


class HtmlParseBybs:
    def __init__(self):
        self.soupstring=None
        pass

    def parse(self,html):
        html = oth_re.sub("", html)
        self.soupstring=BeautifulSoup(html, "html.parser")
        [s.extract() for s in self.soupstring("script")]
        return self

    def get_text(self,tag):
        """
        :param tag:HTML 标签
        :return:解析标签之后的文本内容
        """
        tag_list=[_.get_text() for _ in self.soupstring.select(tag)]
        return tag_list

    def parse_get(self,html,tag):
        """
        :param html:带解析的HTML 字符串
        :param tag: 获取的标签
        :return:  解析标签之后的文本内容
        """
        html=oth_re.sub("",html)
        self.soupstring = BeautifulSoup(html, "html.parser")
        [s.extract() for s in self.soupstring("script")]
        tag_list = [_.get_text() for _ in self.soupstring.select(tag)]
        return tag_list

class HtmlInfo:
    def __init__(self,html):
        self.bsmodel=HtmlParseBybs().parse(html)

    def get_tokenlist(self,tag):
        tmp_list = self.bsmodel.get_text(tag)
        result= [pt_stemmer.stem(w.lower()) for tmp in tmp_list for w in pat_letter.findall(tmp) if(w not in cachedStopWords)]
        return result

    def get_text(self,tag):
        return " ".join(self.bsmodel.get_text(tag))

    def get_sentenceList(self,tag):
        return self.bsmodel.get_text(tag)

    def get_tokendict(self,tag):
        tmp_list = self.bsmodel.get_text(tag)
        tmp_dict = {"doc_lenth":0}
        for tmp in tmp_list:
            tag_list = pat_letter.findall(tmp)
            for token in tag_list:
                if (token in cachedStopWords):
                    continue
                token=pt_stemmer.stem(token.lower())
                tmp_dict[token]=tmp_dict.get(token,0)+1
                tmp_dict["doc_lenth"]=tmp_dict["doc_lenth"]+1
        return tmp_dict
'''

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tqdm import tqdm, trange


batch_size = 1024
trigram_dimension = 30000
noneMatrix = coo_matrix((np.array([]), (np.array([]),np.array([]))), shape=(0, trigram_dimension), dtype=np.int)

value_re=re.compile("[a-zA-Z]")

def get_text(docid,data_path = "/home/user/ntcir_match/en_task/result_html_eng/"):
    file = data_path + "/" + docid + ".html"
    if (not os.path.exists(file)):
        return []
    f = open(file, "rb")
    data = f.read().decode("utf-8", "ignore")
    f.close()
    html_code = "\n".join(data.split("\r\n\r\n")[1:])

    maininfo = HtmlInfo(html_code)
    html_text=maininfo.get_text("html")
    return html_text

def get_value(value):
    return value_re.sub("",value)

def load_queryfile(queryfile):
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
        index_list,value_list = text2wordhashing(content)
        matrix = get_wordhashingMatrix(index_list,value_list)
        query_dict[query_id] = matrix
    return query_dict

def shuffle_data(eval_dict,neg_docs = 4):
    data_list = []
    for key in eval_dict.keys():
        pos_list = []
        neg_list = []
        for docid in eval_dict[key].keys():
            if(eval_dict[key][docid]>0):
                pos_list.append(docid)
            else:
                neg_list.append(docid)
        for posid in pos_list:
            if(len(neg_list)<neg_docs):
                continue
            neglist = random.sample(neg_list, neg_docs)
            data_list.append([key,[posid]+neglist])
    return data_list

def load_baseline(basefile, qIndex=0, subqindex=-1, dIndex=2, split_tag=" "):
    testdata={}
    f = open(basefile, "rt", encoding="utf-8")
    for line in f.readlines():
        datas = line.split(split_tag)
        if (len(datas) < max([qIndex, subqindex, dIndex])):
            continue
        q_id = datas[qIndex]
        if (subqindex != -1):
            q_id = q_id + "_" + datas[subqindex]
        doc_id = datas[dIndex]
        testdata[q_id]=testdata.get(q_id,[])
        testdata[q_id].append(doc_id)
    f.close()
    return testdata

def load_evalfile(similarFile, qIndex=0, subqindex=-1, dIndex=1, simIndex=2, split_tag=" "):
    """
    :param similarFile:人工相似度标签
    :param qIndex: 相似度文档的 query 索引
    :param subqindex: 相似度文档的 subtopic 索引，不存在则为-1
    :param dIndex: 相似度文档的 docid 索引
    :param simIndex: 相似度文档的 相似度 索引
    :param split_tag: 分隔符
    :return:self
    """
    eval_dict = {}
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
        if(int(q_id)>80):
            line = f.readline()
            continue
        doc_id = datas[dIndex]
        sim = get_value(datas[simIndex])
        eval_dict[q_id] = eval_dict.get(q_id, {})
        eval_dict[q_id][doc_id] = int(sim)
        line = f.readline()
    f.close()
    return eval_dict

def get_tokenindex(token):
    index = 0
    for i in range(len(token)):
        s = token[i]
        if(s=="#"):
            index = index + 27*(27**i)
        else:
            value = ord(s)-96
            index = index + value*(27**i)
    return index

def text2trimtoken(text):
    token_list = []
    for word in alpha_re.findall(text.lower()):
        word_str = "#"+word+"#"
        for i in range(len(word_str)-3):
            token = word_str[i:i+3]
            token_list.append(token)
    return token_list
            

def get_wordhashing(token_list):
    index_dict = {}
    index_list = []
    value_list = []
    for token in token_list:
        index = get_tokenindex(token)
        index_dict[index] = index_dict.get(index,len(index_dict.keys()))
        if(index_dict[index]>=len(value_list)):
            value_list.append(0)
        value_list[index_dict[index]] = value_list[index_dict[index]]+1
    index_list = [0]*len(index_dict.keys())
    for key in index_dict.keys():
        index_list[index_dict[key]] = key
    return index_list,value_list

def text2wordhashing(text):
    token_list = text2trimtoken(text)
    index_list,value_list = get_wordhashing(token_list)
    return index_list,value_list

def get_wordhashingMatrix(index_list,value_list):
    _col = np.array([0]*len(index_list))
    _row = np.array(index_list)
    _data = np.array(value_list)
    matrix = coo_matrix((_data, (_col, _row)), shape=(1, trigram_dimension), dtype=np.int)
    return matrix

class DssmForTrainData():
    def __init__(self,train_data,docid_dict,doclenth = 5):
        super(DssmForTrainData, self).__init__()
        self.train_id = train_data
        self.doctext = docid_dict
        self.doclenth = doclenth
    
    def __len__(self):
        return len(self.train_id)

    def __getitem__(self, index):
        q_col = []
        q_row = []
        q_value = []
        qid = self.train_id[index][0]
        qmatrix = self.doctext.get(qid,noneMatrix)
        for i in range(self.doclenth):
            row = qmatrix.row
            value = qmatrix.data
            for row,value in zip(row,value):
                q_col.append(i)
                q_row.append(row)
                q_value.append(value)
        query_tensor = torch.sparse_coo_tensor(torch.tensor([q_col,q_row]), torch.tensor(q_value,dtype = torch.float), (self.doclenth, trigram_dimension))
        _col = []
        _row = []
        _value = []
        for i in range(self.doclenth):
            docid = self.train_id[index][1][i]
            docmatrix = self.doctext.get(docid,noneMatrix)
            row = docmatrix.row
            value = docmatrix.data
            for row,value in zip(row,value):
                _col.append(i)
                _row.append(row)
                _value.append(value)
        doc_tensor = torch.sparse_coo_tensor(torch.tensor([_col,_row]), torch.tensor(_value,dtype = torch.float), (self.doclenth, trigram_dimension))
        target = [0]*self.doclenth
        target[0] = 1
        cos_uni = torch.tensor(target)
        return query_tensor.cuda(),doc_tensor.cuda(),cos_uni.cuda()

class DssmForTestData():
    def __init__(self,qid_list,docid_list,qmatrix,docmatrix):
        super(DssmForTestData, self).__init__()
        self.qid_list = qid_list
        self.docid_list = docid_list
        self.qmatrix = qmatrix
        self.docmatrix = docmatrix
    
    def __len__(self):
        return len(self.qmatrix)

    def __getitem__(self, index):
        q_col = []
        q_row = []
        q_value = []
        qmatrix = self.qmatrix[index]
        row_list = qmatrix.row
        value_list = qmatrix.data
        for row,value in zip(row_list,value_list):
            q_col.append(0)
            q_row.append(row)
            q_value.append(value)
        query_tensor = torch.sparse_coo_tensor(torch.tensor([q_col,q_row]), torch.tensor(q_value,dtype = torch.float), (1, trigram_dimension))
        _col = []
        _row = []
        _value = []
        docmatrix = self.docmatrix[index]
        row_list = docmatrix.row
        value_list = docmatrix.data
        for row,value in zip(row_list,value_list):
            _col.append(0)
            _row.append(row)
            _value.append(value)
        doc_tensor = torch.sparse_coo_tensor(torch.tensor([_col,_row]), torch.tensor(_value,dtype = torch.float), (1, trigram_dimension))
        return self.qid_list[index],self.docid_list[index],query_tensor.to_dense().cuda().view(-1),doc_tensor.to_dense().cuda().view(-1)



class DSSM():
    def __init__(self,batch_size = 8, train_flag=True, model_path = "/home/user/ntcir_match/lzdeep/modelpath/DSSM",neg_lenth = 4):
        self.batch_size = batch_size
        self.model_path = model_path
        self.doclenth = neg_lenth+1
        self.evalmodel = NDCG()
        if (train_flag):
            self.model = dssmNet().cuda()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-5, momentum=0.5)
        else:
            self.model=self.load_model(self.model_path)
        pass

    def train(self,traindata,doctext,epoch=100,max_step=20):
        train_data,val_data = train_test_split(traindata,test_size=0.1)
        train_data = DssmForTrainData(train_data, doctext,doclenth=self.doclenth) 
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
                query_tensor,doc_tensor,cos_uni = batch
                '''
                print(query_tensor)
                print(doc_tensor)
                '''
                query_embed = self.model(query_tensor.to_dense())
                doc_embed = self.model(doc_tensor.to_dense())
                source_cos = torch.cosine_similarity(query_embed, doc_embed, dim=2)
                softmax_qp = torch.nn.functional.softmax(source_cos, dim=1)
                pos_prod = torch.sum(torch.mul(softmax_qp,cos_uni),dim=1)
                #pos_prod = torch.clamp(pos_prod,min=1e-8,max=1)
                loss = -torch.log(torch.prod(pos_prod))

                #acc_tmp = torch.eq(y_.argmax(dim=1),batch_y).sum().float().item()
                #acc_score = acc_score + acc_tmp
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
            qid_list = []
            docid_list = []
            sim_dict = {}
            for qid,doclist in val_data:
                for i in range(len(doclist)):
                    qid_list.append(qid)
                    docid_list.append(doclist[i])
                    sim_dict[qid]=sim_dict.get(qid,{})
                    if(i==0):
                        sim_dict[qid][doclist[i]]=1
                    else:
                        sim_dict[qid][doclist[i]]=0
            result_dict = self.predict(qid_list,docid_list,doctext,test_batchsize=self.batch_size)
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

    def predict(self,qid_list,docid_list,doctext,test_batchsize = 16):
        self.model.eval()
        q_text = []
        doc_text = []
        result_dict = {}
        for i in range(len(qid_list)):
            q_text.append(doctext.get(qid_list[i],""))
            doc_text.append(doctext.get(docid_list[i],""))
        with torch.no_grad():
            test_data = DssmForTestData(qid_list,docid_list,q_text,doc_text)
            test_loader = DataLoader(test_data,batch_size=test_batchsize)
            for batch in test_loader:
                qid,docid,q_text,doc_text = batch
                query_embed = self.model(q_text)
                doc_embed = self.model(doc_text)
                source_cos = torch.cosine_similarity(query_embed, doc_embed, dim=1)
                for qid,docid,cosine in zip(qid,docid,source_cos.tolist()):
                    result_dict[qid] = result_dict.get(qid,{})
                    result_dict[qid][docid] = cosine
        return result_dict
    

class dssmNet(torch.nn.Module):
    def __init__(self):
        super(dssmNet, self).__init__()
        assert (trigram_dimension == 30000)
        self.l1 = torch.nn.Linear(30000, 300)
        torch.nn.init.normal_(self.l1.weight, mean=0, std=1)
        #torch.nn.init.xavier_uniform_(self.l1.weight)
        self.l2 = torch.nn.Linear(300, 300)
        torch.nn.init.normal_(self.l2.weight, mean=0, std=1)
        #torch.nn.init.xavier_uniform_(self.l2.weight)
        self.l3 = torch.nn.Linear(300, 128)
        torch.nn.init.normal_(self.l3.weight, mean=0, std=1)
        #torch.nn.init.xavier_uniform_(self.l3.weight)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x

if __name__=="__main__":
    query_dict = {}
    doctext_dict = {}
    max_doc = 1000
    baseline_path = "/home/user/ntcir_match/en_task/baselineEng.txt"
    htmlfile_path = "/home/user/ntcir_match/en_task/result_html_eng/"
    queryfile="/home/user/ntcir_match/en_task/www2www3topics-E.xml"
    evalfile="/home/user/ntcir_match/en_task/www2e.qrels"
    model_path = "/home/user/ntcir_match/lzdeep/modelpath/DSSM/"
    eval_dict = load_evalfile(evalfile)
    query_dict = load_queryfile(queryfile)
    doctext_dict = query_dict.copy()
    '''
    print("html_text is being loading.....")
    for key in eval_dict.keys():
        for docid in eval_dict[key].keys():
            index_list,value_list = text2wordhashing(get_text(docid))
            matrix = get_wordhashingMatrix(index_list,value_list)
            doctext_dict[docid]=matrix
        print(key+"has been loaded...")
    print("html_text has been loaded.....")
    '''
    train_data = shuffle_data(eval_dict)
    model = DSSM(train_flag=False,model_path = model_path,batch_size=32)
    #model.train(train_data,doctext_dict)
    print("model has been loaded....")
    testdata =load_baseline(baseline_path)
    result_dict = {}
    doctext_dict = query_dict.copy()
    for key in testdata.keys():
        qid_list = []
        docid_list = []
        for docid in testdata[key]:
            qid_list.append(key)
            docid_list.append(docid)
            if(len(docid_list)>=max_doc):
                for docid in docid_list:
                    index_list,value_list = text2wordhashing(get_text(docid))
                    matrix = get_wordhashingMatrix(index_list,value_list)
                    doctext_dict[docid]=matrix
                result = model.predict(qid_list,docid_list,doctext_dict)
                for q in result.keys():
                    result_dict[q]=result_dict.get(q,{})
                    for doc in result[q].keys():
                        result_dict[q][doc]=result[q][doc]
                doctext_dict = query_dict.copy()
                print("1000 file has been computed.....")
        for docid in docid_list:
            index_list,value_list = text2wordhashing(get_text(docid))
            matrix = get_wordhashingMatrix(index_list,value_list)
            doctext_dict[docid]=matrix
        result = model.predict(qid_list,docid_list,doctext_dict)
        for q in result.keys():
            result_dict[q]=result_dict.get(q,{})
            for doc in result[q].keys():
                result_dict[q][doc]=result[q][doc]
        doctext_dict = query_dict.copy()
        print(key+" query has been predicted....")
    
    from trecnorm.normoutput import normTrec
    trec_out=normTrec()
    w = open("/home/user/ntcir_match/lzdeep/drmm/dssm_top1000.txt", "wt", encoding="utf-8")
    for key in result_dict.keys():
        outlines = "\n".join(trec_out.normData(result_dict[key], query_id=key, modelname="DSSM",split_tag=" "))
        w.write(outlines + "\n")
    w.close()
    
