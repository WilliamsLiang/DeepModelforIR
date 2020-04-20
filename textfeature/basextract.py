import math
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec,TaggedDocument

class tfidf:
    def __init__(self):
        self.idf={}
        self.keyData={}

    def load_data(self,data):
        tmp_dict={}
        for key in data.keys():
            word_list=data[key].split(" ")
            for word in word_list:
                tmp_dict[word]=tmp_dict.get(word,[])
                tmp_dict[word].append(key)
                self.keyData[key]=self.keyData.get(key,{})
                self.keyData[key][word]=self.keyData[key].get(word,0)+1
        docs_len=len(data.keys())
        for word in tmp_dict.keys():
            self.idf[word]=math.log(docs_len-len(set(tmp_dict[word]))+0.5)-math.log(len(set(tmp_dict[word]))+0.5)
        for key in self.keyData.keys():
            for word in self.keyData[key].keys():
                self.keyData[key][word]=self.keyData[key][word]*self.idf[word]
        return self

    def extractKeyword(self, data ,n):
        self.load_data(data)
        tmp_dict={}
        for key in self.keyData.keys():
            sort_list=sorted([[word,self.keyData[key][word]] for word in self.keyData[key].keys()],key=lambda x:x[1],reverse=True)
            tmp_dict[key]=[ word for word,_ in sort_list][0:n]
        return tmp_dict

    def get_vector(self):
        pass

class mutualInformation:
    def __init__(self):
        self.prob_c={}
        self.prob_w={}
        self.prob_text={}
        self.keyData={}

    def load_data(self,data):
        for key in data.keys():
            for text in data[key]:
                word_list=text.split(" ")
                self.prob_c[key] = self.prob_c.get(key, 0) + 1
                for word in set(word_list):
                    self.prob_text[key]=self.prob_text.get(key,{})
                    self.prob_text[key][word]=self.prob_text[key].get(word,0)+1
                for word in word_list:
                    self.prob_w[word] = self.prob_w.get(word, 0) + 1
        alltext_num=sum([self.prob_c[key] for key in self.prob_c.keys()])
        allword_num=sum([self.prob_w[key] for key in self.prob_w.keys()])
        for key in self.prob_w.keys():
            self.prob_w[key]=float(self.prob_w[key])/allword_num
        for key in self.prob_c.keys():
            self.keyData[key]={}
            self.prob_c[key]=float(self.prob_c[key])/alltext_num
        for key in self.prob_text.keys():
            for word in self.prob_text[key]:
                self.prob_text[key][word]=float(self.prob_text[key][word])/alltext_num
        for key in self.prob_text.keys():
            for word in self.prob_text[key]:
                tmp_value=self.prob_c[key]*math.log(float(self.prob_text[key][word])/(self.prob_w[word]*self.prob_c[key]),2)
                self.keyData[key][word]=self.keyData[key].get(word,0)
                self.keyData[key][word]=self.keyData[key][word]+tmp_value
        return self

    def extractKeyword(self, data ,n):
        self.load_data(data)
        tmp_dict={}
        for key in self.keyData.keys():
            sort_list=sorted([[word,self.keyData[key][word]] for word in self.keyData[key].keys()],key=lambda x:x[1],reverse=True)
            tmp_dict[key]=[ word for word,_ in sort_list][0:n]
        return tmp_dict

    def get_vector(self):
        pass

class informationGain:
    def __init__(self):
        self.wordClass={}
        self.notwordClass={}
        self.prob_c={}
        self.prob_w={}
        self.keyData={}

    def load_data(self,data):
        for key in data.keys():
            for text in data[key]:
                word_list=text.split(" ")
                self.prob_c[key] = self.prob_c.get(key, 0) + 1
                for word in set(word_list):
                    self.wordClass[word]=self.wordClass.get(word,{})
                    self.wordClass[word][key]= self.wordClass[word].get(key, 0) + 1
                    self.prob_w[word]=self.prob_w.get(word,0)+1
        alltext_num=sum([self.prob_c[key] for key in self.prob_c.keys()])
        allword_num = sum([self.prob_w[key] for key in self.prob_w.keys()])
        for word in self.wordClass.keys():
            w_num=sum([self.wordClass[word][key] for key in self.wordClass[word].keys()])
            for key in self.wordClass[word].keys():
                wc=self.wordClass[word][key]
                w_notc=self.prob_c[key]-self.wordClass[word][key]
                self.wordClass[word][key]=float(wc)/w_num
                self.notwordClass[word]=self.notwordClass.get(word,{})
                self.notwordClass[word][key]=float(w_notc)/self.prob_c[key]
        for key in self.prob_c.keys():
            self.prob_c[key] = float(self.prob_c[key]) / alltext_num
        for key in self.prob_w.keys():
            self.prob_w[key]=float(self.prob_w[key])/allword_num
        for word in self.wordClass.keys():
            self.keyData[word]=0.0
            for key in self.prob_c.keys():
                wc_value=self.wordClass[word].get(key,0) if(self.wordClass[word].get(key,0)) else 0.000000001
                notwc_value=self.notwordClass[word].get(key,0) if(self.notwordClass[word].get(key,0)) else 0.000000001
                self.keyData[word]=self.keyData[word]-self.prob_c[key]*math.log(self.prob_c[key],10)+self.prob_w[word]*wc_value*math.log(wc_value,10)+(1-self.prob_w[word])*notwc_value*math.log(notwc_value,10)
        return self

    def extractKeyword(self, data ,n):
        self.load_data(data)
        sort_list=sorted([[word,self.keyData[word]] for word in self.keyData.keys()],key=lambda x:x[1],reverse=True)
        tmp_list=[ word for word,_ in sort_list][0:n]
        return tmp_list

    def get_vector(self):
        pass

class word2vect_model:
    def __init__(self,path,min_count=2,embedding_dim=64,max_vocab_size=3000,window_size=5):
        self.min_count=min_count
        self.embedding_dim=embedding_dim
        self.max_vocab_size=max_vocab_size
        self.window_size=window_size
        self.modelpath=path

    def train(self,sentences):
        self.model=Word2Vec(sentences,size=self.embedding_dim, window=self.window_size, min_count=self.min_count,max_vocab_size=self.max_vocab_size)
        self.model.save(self.modelpath)
        return self

    def load_model(self):
        self.model=Word2Vec.load(self.modelpath)
        return self

    def get_vector(self,word):
        return self.model[word]

    def is_haveword(self,word):
        try:
            vector=self.model[word]
            return True
        except:
            return False

class doc2vec_model:
    def __init__(self,path,min_count=2,embedding_dim=64,max_vocab_size=3000,window_size=5):
        self.min_count=min_count
        self.embedding_dim=embedding_dim
        self.max_vocab_size=max_vocab_size
        self.window_size=window_size
        self.modelpath=path
        self.vocab={}

    def train(self,docs):
        sentences=[TaggedDocument(docs[key],tags=[key]) for key in docs.keys()]
        self.model=Doc2Vec(sentences,vector_size=self.embedding_dim, window=self.window_size, min_count=self.min_count,
                              max_vocab_size=self.max_vocab_size)
        self.model.save(self.modelpath)
        self.vocab=self.model.vocabulary
        return self

    def load_model(self):
        self.model=Word2Vec.load(self.modelpath)
        self.vocab = self.model.vocabulary
        return self

    def get_vectorByid(self,doc_id):
        return self.model[doc_id]

    def predict_vector(self,sentences):
        return self.model.infer_vector(sentences)

    def is_haveword(self,word):
        tmp_str=self.vocab.get(word,"")
        if(tmp_str):
            return True
        else:
            return False

class keyword_set:
    def __init__(self,path):
        self.keywordset=self.loadkeyword(path)

    def loadkeyword(self,path):
        if(not path):
            return set([])
        word_set=[]
        f = open(path, "rt", encoding="utf-8")
        for line in f.readlines():
            line=line.replace("\r","").replace("\n","")
            if(not line):
                continue
            word_set.append(line)
        f.close()
        return set(word_set)

    def extract_key(self,tmp_data):
        tmp_dict = {}
        for key in tmp_data.keys():
            tmp_dict[key] = [word for word in tmp_data[key].split(" ") if(word in self.keywordset)]
        return tmp_dict

if __name__=="__main__":
    import jieba
    import re


    crime_re = re.compile(r"(.*?罪)、{0,1}")
    docs={}
    model=tfidf()
    f = open("C:/Users/sfe_williamsL/Desktop/毕业论文/result_id.txt", "rt", encoding="utf-8")
    for line in f.readlines():
        datas = line.split("\t")
        if (len(datas) < 2):
            continue
        crimes=crime_re.findall(datas[1])
        content = datas[2]+" "+datas[3]
        for ctype in crimes:
            ctype=ctype.strip("、")
            docs[ctype]=docs.get(ctype,"")
            docs[ctype]=docs[ctype]+" "+" ".join([word for word in jieba.cut(content) if (len(word) > 1)])
            #docs[ctype].append(" ".join([word for word in jieba.cut(content) if (len(word) > 1)]))
    f.close()
    tmp_dict=model.extractKeyword(docs,20)
    #tmp_list=model.extractKeyword(docs,3480)

    w=open("C:/Users/sfe_williamsL/Desktop/keyword_all.txt","wt",encoding="utf-8")
    print(tmp_dict)
    for key in tmp_dict.keys():
        w.write("\n".join(tmp_dict[key])+"\n")
    w.close()


