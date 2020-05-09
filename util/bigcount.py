import re
import os


import nltk.stem.porter as pt
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

from util.basefunction import filter_tags
from util.parsehtml import HtmlParseBybs

pt_stemmer = pt.PorterStemmer()
pat_letter = re.compile(r'[a-zA-Z\-\'\.]+')
cachedStopWords = set(stopwords.words("english"))

class BigCount:
    def __init__(self):
        self.querydict={}
        self.queryset=[]
        self.allword=0
        self.allcorpus=0
        pass

    def loadquery(self,query_file):
        f=open(query_file,"rt",encoding="utf-8")
        xml_text=f.read()
        f.close()
        tmp_dict={}
        soup_string=BeautifulSoup(xml_text, "html.parser")
        query_list=soup_string.select("query")
        for query in query_list:
            query_id=query.select("qid")[0].get_text()
            content=query.select("content")[0].get_text()
            #tmp_dict[query_id] = content.split(" ")
            word_list = pat_letter.findall(content)
            tmp_dict[query_id]=[pt_stemmer.stem(word.lower()) for word in word_list if(word not in cachedStopWords)]
        self.queryset=set([word for key in tmp_dict.keys() for word in tmp_dict[key]])
        return self

    def count(self,filepath,tag):
        file_list=os.listdir(filepath)
        for file in file_list:
            f = open(filepath+file, "rt", encoding="utf-8")
            html = f.read()
            f.close()
            self.count_single(html,tag)
        return self

    def count_single(self,html_code,tag):
        self.allcorpus = self.allcorpus + 1
        tmp_list = HtmlParseBybs().parse_get(html_code, tag)
        tmp_dict = {}
        for tmp in tmp_list:
            tag_list = pat_letter.findall(tmp)
            token_list = [pt_stemmer.stem(w.lower()) for w in tag_list]
            self.allword = self.allword + len(token_list)
            for token in token_list:
                if (token in self.queryset):
                    tmp_dict[token] = tmp_dict.get(token, 0) + 1
        for key in tmp_dict.keys():
            self.querydict[key] = self.querydict.get(key, [0, 0])
            self.querydict[key][0] = self.querydict[key][0] + tmp_dict[key]
            self.querydict[key][1] = self.querydict[key][1] + 1
        return self

    def get_dict(self):
        self.querydict["avg_len"]=[float(self.allword)/self.allcorpus]*2
        self.querydict["all_tokens"]=[self.allword]*2
        self.querydict["all_corpus"]=[self.allcorpus]*2
        return self.querydict


class BigCount_multi:
    def __init__(self,query_file,taglist=["title","body","a","html"]):
        self.modeldict={}
        for key in taglist:
            self.modeldict[key]=BigCount()
            self.modeldict[key].loadquery(query_file)

    def count(self,filepath):
        file_list=os.listdir(filepath)
        for file in file_list:
            f = open(filepath + file, "rt", encoding="utf-8")
            html_code = f.read()
            f.close()
            for key in self.modeldict.keys():
                self.modeldict[key].count_single(html_code,key)
        return self

    def count_single(self,html_code):
        for key in self.modeldict.keys():
            self.modeldict[key].count_single(html_code, key)
        return self

    def get_dict(self,tag):
        return self.modeldict[tag].get_dict()


if __name__=="__main__":
    test_model=BigCount()
    test_model.loadquery("C:/Users/sfe_williamsL/Desktop/任务_待解决/比赛_NTCIR/WWW-3数据/www2www3topics-E.xml")
    print(test_model.queryset)