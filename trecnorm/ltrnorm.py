import re
import os

import nltk.stem.porter as pt

from bs4 import BeautifulSoup
from util.parsehtml import HtmlInfo
from nltk.corpus import stopwords
from simodel.languageModel import BigLanguageModel
from simodel.statisticModel import BigBM25

cachedStopWords = set(stopwords.words("english"))

pt_stemmer = pt.PorterStemmer()
pat_letter = re.compile(r'[a-zA-Z\-\'\.]+')


class ltrdata_generate:
    def __init__(self,queryfile,struct_list=["a","title","body","html"]):
        self.queryfile=queryfile
        self.querydict={}
        self.basedict={}
        self.struct_list=struct_list
        f = open(self.queryfile, "rt", encoding="utf-8")
        xml_text = f.read()
        f.close()
        soup_string = BeautifulSoup(xml_text, "html.parser")
        query_list = soup_string.select("query")
        for query in query_list:
            query_id = query.select("qid")[0].get_text()
            content = query.select("content")[0].get_text()
            # self.querydict[query_id] = content.split(" ")
            word_list = pat_letter.findall(content)
            self.querydict[query_id] = [pt_stemmer.stem(word.lower()) for word in word_list if(word not in cachedStopWords)]
        self.lmdict={}
        self.bm25dict={}
        for key in struct_list:
            self.lmdict[key]=BigLanguageModel(self.querydict)
            self.bm25dict[key]=BigBM25(self.querydict)

    def load_corpus(self,corpuspath):
        for key in self.struct_list:
            self.lmdict[key].loadCorpus(corpuspath+key+"_idf.txt")
            self.bm25dict[key].loadCorpus(corpuspath+key+"_idf.txt")

    def load_baselines(self,basefile,qIndex=0,subqindex=-1,dIndex=2,split_tag=" "):
        f=open(basefile,"rt",encoding="utf-8")
        for line in f.readlines():
            datas=line.split(split_tag)
            if(len(datas)<max([qIndex,subqindex,dIndex])):
                continue
            q_id=datas[qIndex]
            if(subqindex!=-1):
                q_id=q_id+"_"+datas[subqindex]
            doc_id=datas[dIndex]
            self.basedict[doc_id]=self.basedict.get(q_id,{})
            self.basedict[doc_id][q_id]=0
        f.close()
        return self

    def count_doclen(self,doc_dict):
        value=sum([len(doc_dict[key]) for key in doc_dict.keys()])
        return value

    def count_querytf(self,query_token,doc_dict):
        value=0
        for token in query_token:
            value=value+doc_dict.get(token,0)
        return value

    def count_url(self,query_token,baseinfo):
        url=self.get_url(baseinfo)
        value=0
        for token in query_token:
            if(token in url):
                value=value+1
        return value

    def get_url(self,baseinfo):
        url_re=re.compile("X-INKT-URI:.*?\n")
        url_info=" ".join(url_re.findall(baseinfo))
        return url_info

    def outltr(self,filepath,outfile):
        formatline="{score} qid:{qid} {feartures} #docid={docid}\n"
        w=open(outfile,"wt",encoding="utf-8")
        file_list=os.listdir(filepath)
        for file in file_list:
            print(file)
            doc_id=file.replace(".html","")
            f=open(filepath+file,"rb")
            data=f.read().decode("utf-8","ignore")
            f.close()
            html_code="\n".join(data.split("\r\n\r\n")[1:])
            baseinfo=data.split("\r\n\r\n")[0]
            url_value=self.count_url(self.querydict,baseinfo)
            maininfo=HtmlInfo(html_code)
            result_dict={}
            for tag in self.struct_list:
                doc_dict = maininfo.get_tokendict(tag)
                bm25 = self.bm25dict[tag].search(doc_dict)
                lmjm = self.lmdict[tag].search(doc_dict, smooth="JM")
                lmabs = self.lmdict[tag].search(doc_dict, smooth="ABS")
                lmdir = self.lmdict[tag].search(doc_dict, smooth="DIR")
                for key in self.querydict.keys():
                    result_dict[key]=result_dict.get(key,[])
                    tf_value = sum([doc_dict.get(word, 0) for word in self.querydict[key]])
                    idf_dict = self.bm25dict[tag].get_idf()
                    idf_value = sum([idf_dict.get(word, -100) for word in self.querydict[key]])
                    tfidf_value = sum([idf_dict.get(word, -100) * doc_dict.get(word, 0) for word in self.querydict[key]])
                    result_dict[key]=result_dict[key]+[doc_dict["doc_lenth"],tf_value,idf_value,tfidf_value,bm25[key],lmabs[key],lmdir[key],lmjm[key]]
            for key in result_dict:
                result_dict[key].append(url_value)
                fline = self.covertfline(result_dict[key])
                score = self.basedict[doc_id].get(key, -1)
                q_id = key
                outline = formatline.format(score=score, qid=q_id, feartures=fline, docid=doc_id)
                w.write(outline)
                w.flush()
        w.close()

    def covertfline(self,value_list):
        fline_list=[]
        for i in range(len(value_list)):
            fline_list.append(str(i+1)+":"+str(value_list[i]))
        return " ".join(fline_list)


if __name__=="__main__":
    '''
    model=ltrdata_generate("C:/Users/sfe_williamsL/Desktop/任务_待解决/比赛_NTCIR/WWW-3数据/www2www3topics-E.xml")
    model.load_corpus("E:/CLUBWEB_experiment/ntcir_www2/")
    model.load_baselines("C:/Users/sfe_williamsL/Desktop/任务_待解决/比赛_NTCIR/WWW-3数据/baselineEng.txt",qIndex=0,subqindex=-1,dIndex=2,split_tag=" ")
    model.outltr("E:/CLUBWEB_experiment/ntcir_www2/result_html_eng/",outfile="E:/CLUBWEB_experiment/ntcir_www2/ltr_train.txt")
    '''
    docid_re = re.compile(r"docid {0,1}= {0,1}([a-zA-Z0-9\-]*)")
    result=docid_re.findall("docid=clueweb12-0106wb-15-04746 inc = -1 prob = 0.120734")
    print(result)
