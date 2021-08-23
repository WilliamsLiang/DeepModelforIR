import re
import random
import os

import nltk.stem.porter as pt
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

from prepocess.preembedding import embeddingModel,boswModel
from util.parsehtml import HtmlInfo

url_re = re.compile("X-INKT-URI:(.*?)\n")

pt_stemmer = pt.PorterStemmer()
pat_letter = re.compile(r'[a-zA-Z\-\'\.]+')
cachedStopWords = set(stopwords.words("english"))

class trec_data:
    def __init__(self,
                 data_path,
                 query_file,
                 basefile,
                 evalfile,
                 batch_size=16,
                 neg_num=1,
                 embedfile="",
                 option="zeroneg",
                 tag="body",
                 top_rerank=80):
        #变量声明
        self.data_path=data_path
        self.batchsize=batch_size
        self.neg_num=neg_num
        self.evaldict={}
        self.tag=tag
        self.value_re=re.compile("[a-zA-Z]")
        self.train_x=[]
        self.train_q=[]
        self.train_y=[]
        self.val_x=[]
        self.val_y=[]
        self.val_q=[]
        self.test_docid=[]
        self.test_qid=[]
        #函数初始化
        self.query_dict=self.load_queryfile(query_file)
        self.load_evalfile(evalfile)
        self.embedmodel=embeddingModel(embedfile)
        self.load_baseline(basefile,top_num=top_rerank)
        self.shuffle_data(option=option)
        pass

    def load_queryfile(self,queryfile):
        query_dict={}
        f = open(queryfile, "rt", encoding="utf-8")
        xml_text = f.read()
        f.close()
        soup_string = BeautifulSoup(xml_text, "html.parser")
        query_list = soup_string.select("query")
        for query in query_list:
            query_id = query.select("qid")[0].get_text()
            content = query.select("content")[0].get_text()
            # self.querydict[query_id] = content.split(" ")
            word_list = pat_letter.findall(content)
            query_dict[query_id] = [pt_stemmer.stem(word.lower()) for word in word_list if(word not in cachedStopWords)]
        return query_dict

    def load_baseline(self, basefile, qIndex=0, subqindex=-1, dIndex=2, split_tag=" ",top_num=1000):
        tmp_dict={}
        f = open(basefile, "rt", encoding="utf-8")
        for line in f.readlines():
            datas = line.split(split_tag)
            if (len(datas) < max([qIndex, subqindex, dIndex])):
                continue
            q_id = datas[qIndex]
            if (subqindex != -1):
                q_id = q_id + "_" + datas[subqindex]
            doc_id = datas[dIndex]
            tmp_dict[q_id]=tmp_dict.get(q_id,0)+1
            if(tmp_dict[q_id]>top_num):
                continue
            self.test_qid.append(q_id)
            self.test_docid.append(doc_id)
        f.close()
        return self

    def load_evalfile(self,similarFile,qIndex=0,subqindex=-1,dIndex=1,simIndex=2,split_tag=" "):
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

    def generatorTrain(self,doption="seqembeding",qoption="drmm_q",dim=128):
        q_dict={}
        doc_dict={}
        pre_q=""
        for i in range(len(self.train_x)):
            for docid in self.train_x[i]:
                doc_dict[docid]=doc_dict.get(docid,self.get_embedding(self.get_tokenlist(docid),option=doption,dim=dim))
            train_xid=[doc_dict[v] for v in self.train_x[i]]
            qkey=self.train_q[i]
            q_dict[qkey]=q_dict.get(qkey,self.get_embedding(self.get_querytoken(qkey),option=qoption,dim=dim))
            train_xq=q_dict[qkey]
            if(qkey!=pre_q):
                doc_dict={}
                pre_q=qkey
            train_y=self.train_y[i]
            yield train_xq,train_xid,train_y

    def generatorValid(self,doption="seqembeding",qoption="drmm_q",dim=128):
        q_dict={}
        doc_dict={}
        pre_q=""
        for i in range(len(self.val_x)):
            for docid in self.val_x[i]:
                doc_dict[docid]=doc_dict.get(docid,self.get_embedding(self.get_tokenlist(docid),option=doption,dim=dim))
            train_xid = [doc_dict[v] for v in self.val_x[i]]
            qkey=self.val_q[i]
            q_dict[qkey]=q_dict.get(qkey,self.get_embedding(self.get_querytoken(qkey),option=qoption,dim=dim))
            train_xq = self.get_embedding(self.get_querytoken(qkey),option=qoption,dim=dim)
            train_y = self.val_y[i]
            if(qkey!=pre_q):
                doc_dict={}
                pre_q=qkey
            yield train_xq,train_xid,train_y

    def generatorTest(self,doption="seqembeding",qoption="drmm_q",dim=128):
        q_dict={}
        for i in range(len(self.test_qid)):
            test_xid = self.get_embedding(self.get_tokenlist(self.test_docid[i]),option=doption,dim=dim)
            qkey=self.test_qid[i]
            q_dict[qkey]=q_dict.get(qkey,self.get_embedding(self.get_querytoken(qkey),option=qoption,dim=dim))
            test_xq = q_dict[qkey]
            yield self.test_qid[i],self.test_docid[i],test_xid,test_xq

    def get_embedding(self,token_list,option="wordhasing",dim=128):
        if(option=="seqembeding"):
            return self.embedmodel.seqembeding(token_list,dim=dim)
        elif(option=="drmm_q"):
            return self.embedmodel.drmm_qvector(token_list,dim=dim)
        else:
            return self.embedmodel.wordhashing(token_list)

    def get_querytoken(self,queryid):
        return self.query_dict.get(queryid,[])

    def get_tokenlist(self,docid):
        file=self.data_path+"/" + docid+".html"
        if(not os.path.exists(file)):
            return []
        f = open(file, "rb")
        data = f.read().decode("utf-8", "ignore")
        f.close()
        html_code = "\n".join(data.split("\r\n\r\n")[1:])
        baseinfo = data.split("\r\n\r\n")[0]
        url_info = "/".join(url_re.findall(baseinfo)).split("/")
        maininfo = HtmlInfo(html_code)
        token_list = maininfo.get_tokenlist(self.tag)
        all_token = url_info + token_list
        return all_token

    def shuffle_value(self,ratio=0.9,min_num=3):
        data_list = []
        y_list = []
        key_list=[]
        for key in self.evaldict.keys():
            for pos in self.evaldict[key].keys():
                negid = [_ for _ in self.evaldict[key].keys() if (self.evaldict[key][_]<self.evaldict[key][pos])]
                if(len(negid)<self.neg_num):
                    continue
                for i in range(min(min_num,int(len(negid)/self.neg_num))):
                    neglist = random.sample(negid, self.neg_num)
                    data_list.append([pos] + neglist)
                    negy=[self.evaldict[key][_] for _ in neglist]
                    y_list.append([self.evaldict[key][pos]] + negy)
                    key_list.append(key)
        for i in range(len(data_list)):
            if(random.random()<ratio):
                self.train_x.append(data_list[i])
                self.train_y.append(y_list[i])
                self.train_q.append(key_list[i])
            else:
                self.val_x.append(data_list[i])
                self.val_y.append(y_list[i])
                self.val_q.append(key_list[i])
        return self

    def shuffle_zero(self,ratio=0.9):
        data_list=[]
        y_list=[]
        key_list=[]
        for key in self.evaldict.keys():
            posid=[_ for _ in self.evaldict[key].keys() if(self.evaldict[key][_]!=0)]
            negid=[_ for _ in self.evaldict[key].keys() if(self.evaldict[key][_]==0)]
            for pos in posid:
                if(len(negid)<self.neg_num):
                    continue
                neglist=random.sample(negid,self.neg_num)
                data_list.append([pos]+neglist)
                y_list.append([1]+[0]*self.neg_num)
                key_list.append(key)
        for i in range(len(data_list)):
            if (random.random() < ratio):
                self.train_x.append(data_list[i])
                self.train_y.append(y_list[i])
                self.train_q.append(key_list[i])
            else:
                self.val_x.append(data_list[i])
                self.val_y.append(y_list[i])
                self.val_q.append(key_list[i])
        return self

    def shuffle_data(self,option="zeroneg",min_num=1):
        if(option=="valueneg"):
            self.shuffle_value()
        else:
            self.shuffle_zero()



class trec_boswdata:
    def __init__(self,
                 data_path,
                 query_file,
                 basefile,
                 evalfile,
                 batch_size=16,
                 neg_num=1,
                 embedfile="",
                 kmeansfile="",
                 option="zeroneg",
                 tag="body",
                 top_rerank=80):
        #变量声明
        self.data_path=data_path
        self.batchsize=batch_size
        self.neg_num=neg_num
        self.evaldict={}
        self.tag=tag
        self.value_re=re.compile("[a-zA-Z]")
        self.train_x=[]
        self.train_q=[]
        self.train_y=[]
        self.val_x=[]
        self.val_y=[]
        self.val_q=[]
        self.test_docid=[]
        self.test_qid=[]
        #函数初始化
        self.query_dict=self.load_queryfile(query_file)
        self.load_evalfile(evalfile)
        self.embedmodel=boswModel(embedfile,kmeansfile)
        self.load_baseline(basefile,top_num=top_rerank)
        self.shuffle_data(option=option)
        pass

    def load_queryfile(self,queryfile):
        query_dict={}
        f = open(queryfile, "rt", encoding="utf-8")
        xml_text = f.read()
        f.close()
        soup_string = BeautifulSoup(xml_text, "html.parser")
        query_list = soup_string.select("query")
        for query in query_list:
            query_id = query.select("qid")[0].get_text()
            content = query.select("content")[0].get_text()
            # self.querydict[query_id] = content.split(" ")
            word_list = pat_letter.findall(content)
            query_dict[query_id] = [pt_stemmer.stem(word.lower()) for word in word_list if(word not in cachedStopWords)]
        return query_dict

    def load_baseline(self, basefile, qIndex=0, subqindex=-1, dIndex=2, split_tag=" ",top_num=1000):
        tmp_dict={}
        f = open(basefile, "rt", encoding="utf-8")
        for line in f.readlines():
            datas = line.split(split_tag)
            if (len(datas) < max([qIndex, subqindex, dIndex])):
                continue
            q_id = datas[qIndex]
            if (subqindex != -1):
                q_id = q_id + "_" + datas[subqindex]
            doc_id = datas[dIndex]
            tmp_dict[q_id]=tmp_dict.get(q_id,0)+1
            if(tmp_dict[q_id]>top_num):
                continue
            self.test_qid.append(q_id)
            self.test_docid.append(doc_id)
        f.close()
        return self

    def load_evalfile(self,similarFile,qIndex=0,subqindex=-1,dIndex=1,simIndex=2,split_tag=" "):
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

    def generatorTrain(self,doption="docembedding",qoption="query_embedding",dim=128):
        q_dict={}
        doc_dict={}
        pre_q=""
        for i in range(len(self.train_x)):
            for docid in self.train_x[i]:
                doc_dict[docid]=doc_dict.get(docid,self.get_embedding(self.get_tokenlist(docid),option=doption,dim=dim))
            train_xid=[doc_dict[v] for v in self.train_x[i]]
            qkey=self.train_q[i]
            q_dict[qkey]=q_dict.get(qkey,self.get_embedding(self.get_querytoken(qkey),option=qoption,dim=dim))
            train_xq=q_dict[qkey]
            if(qkey!=pre_q):
                doc_dict={}
                pre_q=qkey
            train_y=self.train_y[i]
            yield train_xq,train_xid,train_y


    def generatorValid(self,doption="docembedding",qoption="query_embedding",dim=128):
        q_dict={}
        doc_dict={}
        pre_q=""
        for i in range(len(self.val_x)):
            for docid in self.val_x[i]:
                doc_dict[docid]=doc_dict.get(docid,self.get_embedding(self.get_tokenlist(docid),option=doption,dim=dim))
            train_xid = [doc_dict[v] for v in self.val_x[i]]
            qkey=self.val_q[i]
            q_dict[qkey]=q_dict.get(qkey,self.get_embedding(self.get_querytoken(qkey),option=qoption,dim=dim))
            train_xq = self.get_embedding(self.get_querytoken(qkey),option=qoption,dim=dim)
            train_y = self.val_y[i]
            if(qkey!=pre_q):
                doc_dict={}
                pre_q=qkey
            yield train_xq,train_xid,train_y

    def generatorTest(self,doption="docembedding",qoption="query_embedding",dim=128):
        q_dict={}
        for i in range(len(self.test_qid)):
            test_xid = self.get_embedding(self.get_tokenlist(self.test_docid[i]),option=doption,dim=dim)
            qkey=self.test_qid[i]
            q_dict[qkey]=q_dict.get(qkey,self.get_embedding(self.get_querytoken(qkey),option=qoption,dim=dim))
            test_xq = q_dict[qkey]
            yield self.test_qid[i],self.test_docid[i],test_xid,test_xq

    def get_embedding(self,token_list,option="docembedding",dim=128):
        if(option=="query_embedding"):
            return [token_list,self.embedmodel.clusterpro_vector(token_list,dim=dim)]
        else:
            return self.embedmodel.BOSW_vector(token_list,dim=dim)

    def get_querytoken(self,queryid):
        return self.query_dict.get(queryid,[])

    def get_tokenlist(self,docid):
        file=self.data_path+"/" + docid+".html"
        if(not os.path.exists(file)):
            return []
        f = open(file, "rb")
        data = f.read().decode("utf-8", "ignore")
        f.close()
        html_code = "\n".join(data.split("\r\n\r\n")[1:])
        baseinfo = data.split("\r\n\r\n")[0]
        url_info = "/".join(url_re.findall(baseinfo)).split("/")
        maininfo = HtmlInfo(html_code)
        token_list = maininfo.get_tokenlist(self.tag)
        all_token = url_info + token_list
        return all_token

    def shuffle_value(self,ratio=0.9,min_num=3):
        data_list = []
        y_list = []
        key_list=[]
        for key in self.evaldict.keys():
            for pos in self.evaldict[key].keys():
                negid = [_ for _ in self.evaldict[key].keys() if (self.evaldict[key][_]<self.evaldict[key][pos])]
                if(len(negid)<self.neg_num):
                    continue
                for i in range(min(min_num,int(len(negid)/self.neg_num))):
                    neglist = random.sample(negid, self.neg_num)
                    data_list.append([pos] + neglist)
                    negy=[self.evaldict[key][_] for _ in neglist]
                    y_list.append([self.evaldict[key][pos]] + negy)
                    key_list.append(key)
        for i in range(len(data_list)):
            if(random.random()<ratio):
                self.train_x.append(data_list[i])
                self.train_y.append(y_list[i])
                self.train_q.append(key_list[i])
            else:
                self.val_x.append(data_list[i])
                self.val_y.append(y_list[i])
                self.val_q.append(key_list[i])
        return self

    def shuffle_zero(self,ratio=0.9):
        data_list=[]
        y_list=[]
        key_list=[]
        for key in self.evaldict.keys():
            posid=[_ for _ in self.evaldict[key].keys() if(self.evaldict[key][_]!=0)]
            negid=[_ for _ in self.evaldict[key].keys() if(self.evaldict[key][_]==0)]
            for pos in posid:
                if(len(negid)<self.neg_num):
                    continue
                neglist=random.sample(negid,self.neg_num)
                data_list.append([pos]+neglist)
                y_list.append([1]+[0]*self.neg_num)
                key_list.append(key)
        for i in range(len(data_list)):
            if (random.random() < ratio):
                self.train_x.append(data_list[i])
                self.train_y.append(y_list[i])
                self.train_q.append(key_list[i])
            else:
                self.val_x.append(data_list[i])
                self.val_y.append(y_list[i])
                self.val_q.append(key_list[i])
        return self

    def shuffle_data(self,option="zeroneg",min_num=1):
        if(option=="valueneg"):
            self.shuffle_value()
        else:
            self.shuffle_zero()


class trec_dataRank:
    def __init__(self,
                 data_path,
                 query_file,
                 basefile,
                 evalfile,
                 batch_size=16,
                 neg_num=1,
                 embedfile="",
                 ratio=0.9,
                 tag="body",
                 top_rerank=80):
        #变量声明
        self.data_path=data_path
        self.batchsize=batch_size
        self.neg_num=neg_num
        self.tag=tag
        self.evaldict={}
        self.value_re=self.value_re=re.compile("[a-zA-Z]")
        self.train_x=[]
        self.train_q=[]
        self.train_y=[]
        self.val_x=[]
        self.val_y=[]
        self.val_q=[]
        self.test_docid=[]
        self.test_qid=[]
        #函数初始化
        self.query_dict=self.load_queryfile(query_file)
        self.load_evalfile(evalfile)
        self.embedmodel=embeddingModel(embedfile)
        self.load_baseline(basefile,top_num=top_rerank)
        self.shuffle_rank(ratio=0.9)
        pass

    def load_queryfile(self,queryfile):
        query_dict={}
        f = open(queryfile, "rt", encoding="utf-8")
        xml_text = f.read()
        f.close()
        soup_string = BeautifulSoup(xml_text, "html.parser")
        query_list = soup_string.select("query")
        for query in query_list:
            query_id = query.select("qid")[0].get_text()
            content = query.select("content")[0].get_text()
            # self.querydict[query_id] = content.split(" ")
            word_list = pat_letter.findall(content)
            query_dict[query_id] = [pt_stemmer.stem(word.lower()) for word in word_list if(word not in cachedStopWords)]
        return query_dict

    def load_baseline(self, basefile, qIndex=0, subqindex=-1, dIndex=2, split_tag=" ",top_num=80):
        tmp_dict={}
        f = open(basefile, "rt", encoding="utf-8")
        for line in f.readlines():
            datas = line.split(split_tag)
            if (len(datas) < max([qIndex, subqindex, dIndex])):
                continue
            q_id = datas[qIndex]
            if (subqindex != -1):
                q_id = q_id + "_" + datas[subqindex]
            doc_id = datas[dIndex]
            tmp_dict[q_id]=tmp_dict.get(q_id,0)+1
            if(tmp_dict[q_id]>top_num):
                continue
            self.test_qid.append(q_id)
            self.test_docid.append(doc_id)
        f.close()
        return self

    def load_evalfile(self,similarFile,qIndex=0,subqindex=-1,dIndex=1,simIndex=2,split_tag=" "):
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

    def generatorTrain(self,doption="seqembeding",qoption="drmm_q",dim=128):
        q_dict={}
        doc_dict={}
        pre_q=""
        for i in range(len(self.train_x)):
            doc_id=self.train_x[i]
            doc_dict[doc_id]=doc_dict.get(doc_id,self.get_embedding(self.get_tokenlist(doc_id),option=doption,dim=dim))
            train_xid=doc_dict[doc_id]
            qkey=self.train_q[i]
            q_dict[qkey]=q_dict.get(qkey,self.get_embedding(self.get_querytoken(qkey),option=qoption,dim=dim))
            train_xq=q_dict[qkey]
            if(qkey!=pre_q):
                doc_dict={}
                pre_q=qkey
            train_y=self.train_y[i]
            yield train_xq,train_xid,train_y


    def generatorValid(self,doption="seqembeding",qoption="drmm_q",dim=128):
        q_dict={}
        doc_dict={}
        pre_q=""
        for i in range(len(self.val_x)):
            doc_id=self.val_x[i]
            doc_dict[doc_id]=doc_dict.get(doc_id,self.get_embedding(self.get_tokenlist(doc_id),option=doption,dim=dim))
            train_xid = doc_dict[doc_id]
            qkey=self.val_q[i]
            q_dict[qkey]=q_dict.get(qkey,self.get_embedding(self.get_querytoken(qkey),option=qoption,dim=dim))
            train_xq = self.get_embedding(self.get_querytoken(qkey),option=qoption,dim=dim)
            train_y = self.val_y[i]
            if(qkey!=pre_q):
                doc_dict={}
                pre_q=qkey
            yield train_xq,train_xid,train_y

    def generatorTest(self,doption="seqembeding",qoption="drmm_q",dim=128):
        q_dict={}
        for i in range(len(self.test_qid)):
            test_xid = self.get_embedding(self.get_tokenlist(self.test_docid[i]),option=doption,dim=dim)
            qkey=self.test_qid[i]
            q_dict[qkey]=q_dict.get(qkey,self.get_embedding(self.get_querytoken(qkey),option=qoption,dim=dim))
            test_xq = q_dict[qkey]
            yield self.test_qid[i],self.test_docid[i],test_xid,test_xq

    def get_embedding(self,token_list,option="wordhasing",dim=128):
        if(option=="seqembeding"):
            return self.embedmodel.seqembeding(token_list,dim=dim)
        elif(option=="drmm_q"):
            return self.embedmodel.drmm_qvector(token_list,dim=dim)
        else:
            return self.embedmodel.wordhashing(token_list)

    def get_querytoken(self,queryid):
        return self.query_dict.get(queryid,[])

    def get_tokenlist(self,docid):
        file=self.data_path+"/" + docid+".html"
        if(not os.path.exists(file)):
            return []
        f = open(file, "rb")
        data = f.read().decode("utf-8", "ignore")
        f.close()
        html_code = "\n".join(data.split("\r\n\r\n")[1:])
        baseinfo = data.split("\r\n\r\n")[0]
        url_info = "/".join(url_re.findall(baseinfo)).split("/")
        maininfo = HtmlInfo(html_code)
        token_list = maininfo.get_tokenlist(self.tag)
        all_token = url_info + token_list
        return all_token

    def shuffle_rank(self,ratio=0.9):
        data_list = []
        y_list = []
        key_list=[]
        for key in self.evaldict.keys():
            for pos in self.evaldict[key].keys():
                data_list.append(pos)
                y_list.append(self.evaldict[key][pos])
                key_list.append(key)
        for i in range(len(data_list)):
            if(random.random()<ratio):
                self.train_x.append(data_list[i])
                self.train_y.append(y_list[i])
                self.train_q.append(key_list[i])
            else:
                self.val_x.append(data_list[i])
                self.val_y.append(y_list[i])
                self.val_q.append(key_list[i])
        return self

if __name__=="__main__":
    data_model=trec_data(data_path="E:/CLUBWEB_experiment/ntcir_www2/result_html_eng",
                 query_file="D:/任务_待解决/比赛_NTCIR/WWW-3数据/www2www3topics-E.xml",
                 evalfile="D:/任务_待解决/比赛_NTCIR/WWW-3数据/www2e.qrels",
                 batch_size=16,
                 neg_num=1,
                 embedfile="")
    data_model.shuffle_data(option="valueneg")
    for x,y,z in data_model.generatorTrain():

        print(x)
        print(y)
        print(z)

        print("------------")

