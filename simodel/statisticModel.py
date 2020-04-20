import math
import json


import jieba
from textfeature.basextract import tfidf

class BM25():
    def __init__(self,docs={},isLoadpath=""):
        self.k1 = 1.5
        self.b = 0.75
        self.D=0
        self.dictDj = {}
        self.indexdict = {}
        self.idf = {}
        self.avgdl=0
        if(isLoadpath):
            self.loadIndexdata(isLoadpath)
        else:
            self.D = len(docs)
            sumdlen=0
            for doc_id in docs.keys():
                singlen=len(set(docs[doc_id]))
                sumdlen=sumdlen+singlen
                self.dictDj[doc_id]=singlen
            self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / sumdlen
            self.init(docs)
        pass

    #生成倒排索引和词汇的IDF值
    def init(self,docs):
        for doc_id in docs.keys():
            for word in docs[doc_id]:
                self.indexdict[word]=self.indexdict.get(word,{})
                self.indexdict[word][doc_id]=self.indexdict[word].get(doc_id,0)+1
        for word in self.indexdict.keys():
            self.idf[word] = math.log(self.D-len(self.indexdict[word])+0.5)-math.log(len(self.indexdict[word])+0.5)

    def queryWord(self, word):
        tmp_result={}
        docid_list=self.indexdict.get(word,{}).keys()
        for docid in docid_list:
            k_score = (self.idf[word] * self.indexdict[word][docid] * (self.k1 + 1)/ (self.indexdict[word][docid] + self.k1 * (1 - self.b + self.b * self.dictDj[docid]/ self.avgdl)))
            tmp_result[docid]=k_score
        return tmp_result

    def queryWordset(self, query_list):
        scores={}
        for query in  query_list:
            tmp_result=self.queryWord(query)
            for docid in tmp_result.keys():
                scores[docid]=scores.get(docid,0)+tmp_result[docid]
        return scores

    def dumpIndexdata(self,path):
        tmp_dict={}
        tmp_dict["docJ"]=self.dictDj
        tmp_dict["index"]=self.indexdict
        tmp_dict["idf"]=self.idf
        tmp_dict["avgdl"] = self.avgdl
        tmp_dict["lenD"] = self.D
        json_str=json.dumps(tmp_dict)
        with open(path,"wt",encoding="utf-8") as w:
            w.write(json_str)


    def loadIndexdata(self,path):
        with open(path, "rt", encoding="utf-8") as f:
            line=f.readline()
        tmp_dict=json.loads(line)
        self.dictDj=tmp_dict["docJ"]
        self.indexdict=tmp_dict["index"]
        self.idf=tmp_dict["idf"]
        self.avgdl=tmp_dict["avgdl"]
        self.D=tmp_dict["lenD"]


if __name__=="__main__":
    docs = {}
    '''
    f = open("C:/Users/sfe_williamsL/Desktop/毕业论文/result_id.txt", "rt", encoding="utf-8")
    for line in f.readlines():
        datas = line.split("\t")
        if (len(datas) < 2):
            continue
        docid = datas[0]
        content = datas[3]
        docs[docid] = list(jieba.cut(content))
    f.close()
    '''
    BM25model = BM25(docs, isLoadpath="C:/Users/sfe_williamsL/Desktop/毕业论文/BM25_short_data.json")
    print("模型预加载完毕")
    from trecnorm.normoutput import normTrec

    trec_out = normTrec()
    f = open("C:/Users/sfe_williamsL/Desktop/毕业论文/data/test.txt", "rt", encoding="utf-8")
    tmp_data = {}
    for line in f.readlines():
        datas = line.split("\t")
        if (len(datas) < 2):
            continue
        query_id = datas[0]
        content = datas[4]
        query_word = " ".join(jieba.cut(content))
        tmp_data[query_id] = query_word
    f.close()
    w = open("C:/Users/sfe_williamsL/Desktop/毕业论文/base_query/BM25query_IG.txt", "wt", encoding="utf-8")
    from textfeature.basextract import keyword_set
    keymodel=keyword_set("C:/Users/sfe_williamsL/Desktop/毕业论文/keyword_short_50_IG.txt")
    query_data = keymodel.extract_key(tmp_data)
    #query_data = tfidf().extractKeyword(tmp_data, 50)
    # print(query_data)
    for key in query_data.keys():
        results = BM25model.queryWordset(query_data[key])
        outlines = "\n".join(trec_out.normData(results, query_id=key, modelname="BM25"))
        w.write(outlines + "\n")
    w.close()
    BM25model.dumpIndexdata("C:/Users/sfe_williamsL/Desktop/毕业论文/BM25_short_data.json")