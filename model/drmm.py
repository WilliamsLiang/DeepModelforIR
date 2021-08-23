import torch
import math
import numpy as np
import os 
import numpy as np
import re
import nltk.stem.porter as pt

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm, trange

from prepocess.preembedding import embeddingModel
from util.parsehtml import HtmlInfo
from util.basefunction import load_evalfile,shuffle_rank,load_baseline

url_re = re.compile("X-INKT-URI:(.*?)\n")

pt_stemmer = pt.PorterStemmer()
pat_letter = re.compile(r'[a-zA-Z\-\'\.]+')
cachedStopWords = set(stopwords.words("english"))

class DrmmForTrainData():
    def __init__(self,traindata,idfdata):
        super(DrmmForTrainData, self).__init__()
        self.train_data = traindata
        self.idf_data = idfdata
    
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        pos_embed = self.train_data[index][0]
        neg_embed = self.train_data[index][1]
        pos_tensor = torch.tensor(pos_embed,dtype=torch.float32)
        neg_tensor = torch.tensor(neg_embed,dtype=torch.float32)
        idf_tensor = torch.tensor(self.idf_data[index],dtype=torch.float32)
        return pos_tensor.cuda(),neg_tensor.cuda(),idf_tensor.cuda()

class DrmmForTestData():
    def __init__(self,idlist,traindata,idfdata):
        super(DrmmForTestData, self).__init__()
        self.idlist = idlist
        self.test_data = traindata
        self.idf_data = idfdata
    
    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        qid = self.idlist[index][0]
        docid = self.idlist[index][1]
        rel_embed = self.test_data[index]
        rel_tensor = torch.tensor(rel_embed,dtype=torch.float32)
        idf_tensor = torch.tensor(self.idf_data[index],dtype=torch.float32)
        return qid,docid,rel_tensor.cuda(),idf_tensor.cuda()

class DrmmDataloader():
    def __init__(self,
                 bin=0.5,
                 querynum=5,
                 idf_file="",
                 embedfile="",
                 queryfile="",
                 data_path = "",
                 tag_name = "body",
                 wc_index=2
                ):
        self.bin=bin
        self.binum=int(2/bin)+1
        self.max_num=querynum
        self.data_path = data_path
        self.idf={}
        self.tag = tag_name
        self.loadCorpus(idf_file,wcindex=wc_index)
        self.embedmodel=embeddingModel(embedfile)
        self.loadCorpus(idf_file)
        self.load_queryfile(queryfile)
        self.cache_dict = {}
    
    def loadCorpus(self,idf_file,wcindex=2):
        """
        :param Corpus_file: 计算好的文件，含p(w|c),文档集tokens数
        :return:
        """
        tmp_dict={}
        f=open(idf_file,"rt",encoding="utf=8")
        for line in f.readlines():
            datas=line.split("\t")
            if(len(datas)<2):
                continue
            key=datas[0]
            value=datas[wcindex]
            tmp_dict[key]=float(value)
        f.close()
        for key in tmp_dict.keys():
            if (key == "avg_len"):
                continue
            elif (key == "all_tokens"):
                continue
            elif (key == "all_corpus"):
                continue
            self.idf[key]=math.log(tmp_dict["all_corpus"] - tmp_dict[key] + 0.5) - math.log(tmp_dict[key] + 0.5)
        return self
    
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
            word_list = pat_letter.findall(content)
            self.query_dict[query_id] = [pt_stemmer.stem(word.lower()) for word in word_list if(word not in cachedStopWords)]
        return self

    def get_histogramMatrix(self,q_vector,d_vecotr):
        histogram=[]
        i=0
        for m in range(min(len(q_vector),self.max_num)):
            query = q_vector[m]
            bin_vector=self.get_histogram(query,d_vecotr)
            histogram.append(bin_vector)
            i=i+1
        for j in range(i,self.max_num):
            histogram.append([1]*self.binum)
        return histogram

    def get_histogram(self,word,d_vector):
        bin_vector=[1]*self.binum
        for vector in d_vector:
            cos_value=torch.cosine_similarity(torch.Tensor(word), torch.Tensor(vector), dim=0)
            index=int((cos_value+1)/self.bin)
            bin_vector[index]=bin_vector[index]+1
        return bin_vector

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

    def get_idf(self,token_list):
        token_weight=[]
        for token in token_list:
            value=self.idf.get(token,10)
            token_weight.append(value)
        token_weight = token_weight[0:self.max_num]
        minus=self.max_num-len(token_weight)
        if(minus<1):
            minus=0
        return token_weight+[0]*minus
    
    def text2vector(self,qid,docid,dim=128):
        query_token = self.get_querytoken(qid)
        if(self.cache_dict.get(qid,{}).get(docid,[])):
            return self.cache_dict[qid][docid]
        doc_token = self.get_tokenlist(docid)
        query_embedding = self.get_embedding(query_token,option="drmm_q",dim=dim)
        doc_embedding = self.get_embedding(doc_token,option="seqembeding",dim=dim)
        histogram_matrix = self.get_histogramMatrix(query_embedding,doc_embedding)
        self.cache_dict[qid] = self.cache_dict.get(qid,{})
        self.cache_dict[qid][docid] = histogram_matrix
        return self.cache_dict[qid][docid]

    def getIdfvector(self,qid):
        query_token = self.get_querytoken(qid)
        idf_vector = self.get_idf(query_token)
        return idf_vector
    
    def cache_clear(self):
        self.cache_dict = {}

class DRMM():
    def __init__(self,
                 bin=0.5,
                 querynum=5,
                 learning_rate=0.01,
                 train_flag=True,
                 batch_size = 8,
                 model_path = "/home/user/ntcir_match/lzdeep/modelpath/DRRM/"):
        self.bin=bin
        self.binum=int(2/bin)+1
        self.batch_size = batch_size
        self.model_path = model_path
        if(train_flag):
            self.model = drmmNet(histogram_num=self.binum, query_num=querynum).cuda()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.model = self.load_model(model_path)
        pass

    def train(self,data,drmmloader,epoch=100,max_step=20):
        traindata,valdata = train_test_split(data,test_size=0.1)
        train_data = []
        idf_data = []
        tag_num = 0
        for qid,docidlist in traindata:
            tmp_data = []
            idf_vector = drmmloader.getIdfvector(qid)
            for docid in docidlist:
                histogram = drmmloader.text2vector(qid,docid)
                tmp_data.append(histogram)
            train_data.append(tmp_data)
            idf_data.append(idf_vector)
            tag_num = tag_num + 1 
            if(tag_num%1000==0):
                print(str(tag_num) + " has been loaded....")
        train_data = DrmmForTrainData(train_data,idf_data) 
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
                pos_tensor,neg_tensor,idf_tensor = batch
                pos_value = self.model(pos_tensor, idf_tensor)
                neg_value = self.model(neg_tensor, idf_tensor)
                
                criterion = DrmmLoss()
                loss=criterion(pos_value,neg_value)
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
            val_negdata = []
            for qid,docidlist in valdata:
                pos_id = docidlist[0]
                neg_id = docidlist[1]
                val_posdata.append([qid,pos_id])
                val_negdata.append([qid,neg_id])

            result_pos = self.predict(val_posdata,drmmloader,test_batchsize=self.batch_size)
            result_neg = self.predict(val_negdata,drmmloader,test_batchsize=self.batch_size)

            val_result = []
            for qid,docidlist in valdata:
                pos_id = docidlist[0]
                neg_id = docidlist[1]
                val_result.append([result_pos[qid][pos_id],result_neg[qid][neg_id]])
            acc_final = [0]*len(val_result)
            mean_acc = torch.eq(torch.tensor(val_result,dtype = torch.float32).argmax(dim=1),torch.tensor(acc_final)).sum().float().item()/len(val_result)
            stop_batch = stop_batch + 1
            #print(test_y)
            print("- train_data loss /{:04.2f}/ - dev_data acc /{:04.2f}/".format(all_loss , mean_acc))
            tqdm_bar.close()
            if (mean_acc > pre_value):
                pre_value = mean_acc
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

    def predict(self,data,drmmloader,test_batchsize = 16):
        result_dict = {}
        test = []
        idf = []
        for qid,docid in data:
            idf_vector = drmmloader.getIdfvector(qid)
            histogram  = drmmloader.text2vector(qid,docid)
            test.append(histogram)
            idf.append(idf_vector)
        self.model.eval()
        with torch.no_grad():
            test_data = DrmmForTestData(data,test,idf)
            test_loader = DataLoader(test_data,batch_size=test_batchsize)
            for batch in test_loader:
                qid,docid,histogram_tensor,idf_tensor = batch
                rel_value = self.model(histogram_tensor, idf_tensor)
                for qid,docid,cosine in zip(qid,docid,rel_value.tolist()):
                    result_dict[qid] = result_dict.get(qid,{})
                    result_dict[qid][docid] = cosine
        return result_dict

class DrmmLoss(torch.nn.Module):
    def _init_(self):
        super(DrmmLoss, self).__init__()

    def forward(self, pos_value, neg_value):
        sub = torch.sub(pos_value,neg_value)
        #loss = torch.mean(torch.max(torch.tensor([0.0]).cuda(),torch.sub(1,sub)))
        loss=torch.mean(torch.log(torch.add(1.0,torch.exp(torch.mul(sub,-1.6)))))
        return loss

class drmmNet(torch.nn.Module):
    def __init__(self,histogram_num=5,query_num=5):
        super(drmmNet, self).__init__()
        self.max_query_word=query_num
        self.max_bin_size=histogram_num
        self.histogram_dense_1=torch.nn.Linear(histogram_num, 3)
        torch.nn.init.normal_(self.histogram_dense_1.weight,mean=0,std=4.0)
        self.query_term_dense=torch.nn.Linear(3,1)
        torch.nn.init.normal_(self.query_term_dense.weight,mean=0,std=4.0)
        self.weight = torch.randn([1],requires_grad=True,device="cuda")

    def forward(self, histogram,idf):
        histogram = torch.log10(histogram)
        hidden_1 = torch.nn.functional.relu(self.histogram_dense_1(histogram))
        query_term_score = torch.nn.functional.relu(self.query_term_dense(hidden_1))
        #self.query_term_score = torch.reshape(query_term_score, [-1, self.max_query_word])
        #add gating term
        self.term_gate = torch.div(torch.exp(torch.mul(idf,self.weight)),torch.sum(torch.exp(torch.mul(idf,self.weight))))
        #matmul two tensors
        m,n,_ = query_term_score.size()
        self.query_term_score=torch.reshape(query_term_score,(m,n))
        
        self.match_score=torch.sum(torch.mul(self.query_term_score,self.term_gate),dim=1)
        return self.match_score 

if __name__=="__main__":
    query_dict = {}
    doctext_dict = {}
    max_doc = 1000
    baseline_path = "/home/user/ntcir_match/en_task/baselineEng.txt"
    htmlfile_path = "/home/user/ntcir_match/en_task/result_html_eng/"
    queryfile="/home/user/ntcir_match/en_task/www2www3topics-E.xml"
    evalfile="/home/user/ntcir_match/en_task/www2e.qrels"
    model_path = "/home/user/ntcir_match/lzdeep/modelpath/DSSM/"
    idf_file="/home/user/ntcir_match/en_task/html_idf.txt"
    embedfile="/home/user/ntcir_match/lzdeep/word2vec/word2vec.model"
    data_loader = DrmmDataloader(bin=0.5,
                 querynum=5,
                 idf_file=idf_file,
                 embedfile=embedfile,
                 queryfile=queryfile,
                 data_path = htmlfile_path,
                 wc_index=2)

    eval_dict = load_evalfile(evalfile)
    
    train_data = shuffle_rank(eval_dict)
    model = DRMM(train_flag=False,model_path = model_path,batch_size=32)
    model.train(train_data,data_loader)
    print("model has been loaded....")
    testdata =load_baseline(baseline_path)
    result_dict = {}
    for key in testdata.keys():
        test_data = []
        for docid in testdata[key]:
            test_data.append([key,docid])
            if(len(test_data)>=max_doc):
                
                result = model.predict(test_data,data_loader)
                for q in result.keys():
                    result_dict[q]=result_dict.get(q,{})
                    for doc in result[q].keys():
                        result_dict[q][doc]=result[q][doc]
                data_loader.cache_clear()
                print("1000 file has been computed.....")
        result = model.predict(test_data,data_loader)
        for q in result.keys():
            result_dict[q]=result_dict.get(q,{})
            for doc in result[q].keys():
                result_dict[q][doc]=result[q][doc]
        data_loader.cache_clear()
        print(key+" query has been predicted....")
    
    from trecnorm.normoutput import normTrec
    trec_out=normTrec()
    w = open("/home/user/ntcir_match/lzdeep/drmm/drmm_top1000.txt", "wt", encoding="utf-8")
    for key in result_dict.keys():
        outlines = "\n".join(trec_out.normData(result_dict[key], query_id=key, modelname="DRMM",split_tag=" "))
        w.write(outlines + "\n")
    w.close()
    