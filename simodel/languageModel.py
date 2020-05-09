import math
import json


class LanguageModel:
    def __init__(self, corpus, lamb=0.1, mu=2000, delta=0.7):
        """
        Use language models to score query/document pairs.
        :param corpus:
        :param lamb:
        :param mu:
        :param delta:
        """
        self.lamb = lamb
        self.mu = mu
        self.delta = delta

        # Fetch all of the necessary quantities for the document language
        # models.
        doc_token_counts = []
        doc_lens = [] #后续计算
        doc_p_mls = [] #后续计算公式c(w,d)
        all_token_counts = {} #加载向量 p(w|c)
        for doc in corpus:
            doc_len = len(doc)
            doc_lens.append(doc_len)
            token_counts = {}
            for token in doc:
                token_counts[token] = token_counts.get(token, 0) + 1
                all_token_counts[token] = all_token_counts.get(token, 0) + 1

            doc_token_counts.append(token_counts)

            p_ml = {}
            for token in token_counts:
                p_ml[token] = token_counts[token] / doc_len

            doc_p_mls.append(p_ml)

        total_tokens = sum(all_token_counts.values())
        p_C = {
            token: token_count / total_tokens
            for (token, token_count) in all_token_counts.items()
        }

        self.N = len(corpus) #加载文件  这个值在此公式可以忽略 BM25需要
        self.c = doc_token_counts # 文档中 token的数量
        self.doc_lens = doc_lens # 文档中 词的数量 C
        self.p_ml = doc_p_mls #文档中 p(w|d) 后续计算 c(w,d)
        self.p_C = p_C #加载向量 p(w|c)

    def jelinek_mercer(self, query_tokens):
        """
        Calculate the Jelinek-Mercer scores for a given query.
        :param query_tokens:
        :return:
        """
        lamb = self.lamb
        p_C = self.p_C
        scores = {}
        for doc_idx in range(self.N):
            p_ml = self.p_ml[doc_idx]
            score = 0
            for token in query_tokens:
                if token not in p_C:
                    continue

                score -= math.log((1 - lamb) * p_ml.get(token, 0) + lamb * p_C[token])
            scores[doc_idx]=scores
        return scores

    def dirichlet(self, query_tokens):
        """
        Calculate the Dirichlet scores for a given query.
        :param query_tokens:
        :return:
        """
        mu = self.mu
        p_C = self.p_C

        scores = {}
        for doc_idx in range(self.N):
            c = self.c[doc_idx]
            doc_len = self.doc_lens[doc_idx]
            score = 0
            for token in query_tokens:
                if token not in p_C:
                    continue
                score -= math.log((c.get(token, 0) + mu * p_C[token]) / (doc_len + mu))
            scores[doc_idx] = scores
        return scores

    def absolute_discount(self, query_tokens):
        """
        Calculate the absolute discount scores for a given query.
        :param query_tokens:
        :return:
        """
        delta = self.delta
        p_C = self.p_C

        scores = {}
        for doc_idx in range(self.N):
            c = self.c[doc_idx]
            doc_len = self.doc_lens[doc_idx]
            d_u = len(c)
            score = 0
            for token in query_tokens:
                if token not in p_C:
                    continue
                score -= math.log(
                    max(c.get(token, 0) - delta, 0) / doc_len
                    + delta * d_u / doc_len * p_C[token]
                )
            scores[doc_idx] = scores
        return scores


class BigLanguageModel:
    def __init__(self,query_token,lamb=0.1, mu=2000, delta=0.7):
        """
        Use language models to score query/document pairs.
        目的解决大数据环境下内存不足的问题，仍然存在速度慢的问题，建议使用mapreduce的方式进行计算。
        :param corpus:
        :param lamb:
        :param mu:
        :param delta:
        """
        self.lamb = lamb
        self.mu = mu
        self.delta = delta
        self.p_C = {}  # 加载向量 p(w|c)
        self.query_token=query_token
        #self.loadCorpus(Corpus_file)

    def loadCorpus(self,Corpus_file,wcindex=1):
        """
        :param Corpus_file: 计算好的文件，含p(w|c),文档集tokens数
        :return:
        """
        tmp_dict={}
        f=open(Corpus_file,"rt",encoding="utf=8")
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
                self.avgdl = tmp_dict[key]
                continue
            elif (key == "all_tokens"):
                continue
            elif(key=="all_corpus"):
                continue
            self.p_C[key] = tmp_dict[key] / tmp_dict["all_tokens"]
        return self

    def select_method(self,smooth):
        if(smooth=="ABS"):
            return self.absolute_discount
        elif(smooth=="DIR"):
            return self.dirichlet
        else:
            return self.jelinek_mercer

    def search(self,doc_dict,smooth="JM"):
        relmodel=self.select_method(smooth)
        return relmodel(doc_dict)

    def search_text(self,doc_text,smooth="JM"):
        doc_dic={}
        doc_dic["doc_lenth"]=len(doc_text)
        for word in doc_text:
            doc_dic[word]=doc_dic.get(word,0)+1
        return self.search(doc_dic,smooth=smooth)

    def jelinek_mercer(self, doc_dict):
        tmp_dict={}
        for key in self.query_token.keys():
            lamb=self.lamb
            doc_len = doc_dict.get("doc_lenth",0) + 1
            score = 0
            for token in self.query_token[key]:
                if(self.p_C.get(token,0)==0):
                    continue
                score -= math.log((1 - lamb) * (doc_dict.get(token,0.0)/doc_len) + lamb * self.p_C[token])
            tmp_dict[key]=score
        return tmp_dict

    def dirichlet(self, doc_dict):
        tmp_dict={}
        mu = self.mu
        for key in self.query_token.keys():
            doc_len = doc_dict.get("doc_lenth",0)+1
            score = 0
            for token in self.query_token[key]:
                if(self.p_C.get(token, 0) == 0):
                    continue
                score -= math.log((doc_dict.get(token, 0) + mu * self.p_C[token]) / (doc_len + mu))
            tmp_dict[key] = score
        return tmp_dict

    def absolute_discount(self, doc_dict):
        tmp_dict={}
        delta = self.delta
        for key in self.query_token.keys():
            doc_len = doc_dict.get("doc_lenth",0)+1
            d_u = len(doc_dict.keys())
            score = 0
            for token in self.query_token[key]:
                if (self.p_C.get(token, 0) == 0):
                    continue
                score -= math.log(max(doc_dict.get(token, 0) - delta, 0) / doc_len  + delta * d_u / doc_len * self.p_C[token])
            tmp_dict[key] = score
        return tmp_dict
