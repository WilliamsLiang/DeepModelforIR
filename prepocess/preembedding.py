import gensim
from gensim.models import Word2Vec
import numpy as np
import scipy
import joblib

class embeddingModel:
    def __init__(self,vectorfile):
        self.model=Word2Vec.load(vectorfile)

    def seqembeding(self,token_list,dim=300):
        seq_vector=[]
        for token in token_list:
            embed=self.get_vector(token,dim=dim)
            seq_vector.append(embed)
        return seq_vector

    def drmm_qvector(self,token_list,dim=300):
        q_vector=[]
        for token in token_list:
            embed=self.get_vector(token,dim=dim)
            q_vector.append(embed)
        return q_vector

    def get_embedding(self,token_list,option="seq",dim=300):
        if(option=="drmm_qvctor"):
            return self.drmm_qvector(token_list,dim=dim)
        else:
            return self.seqembeding(token_list,dim=dim)

    def wordhashing(self,token_list):
        return []

    def get_vector(self,word,dim=300):
        try:
            return self.model[word].tolist()
        except:
            return [0]*dim


class boswModel:
    def __init__(self,vectorfile,kmeans_file,cluster_num=50):
        self.model=Word2Vec.load(vectorfile)
        self.clustermodel=joblib.load(kmeans_file)
        self.cluster_num=self.clustermodel.get_params().get("n_clusters",cluster_num)

    def BOSW_vector(self,token_list,dim=300):
        bosw_vector=[0]*self.cluster_num
        for token in token_list:
            embed=self.get_vector(token,dim=dim)
            index=self.clustermodel.predict([embed])[0]
            bosw_vector[index]=bosw_vector[index]+1
        return np.array(bosw_vector)

    def clusterpro_vector(self,token_list,dim=300):
        cluster_vector=[]
        for token in token_list:
            embed=self.get_vector(token,dim=dim)
            weight_embed=self.clustermodel.transform([embed])[0]
            max_minus=max(weight_embed.tolist())-min(weight_embed.tolist())
            max_value=max(weight_embed.tolist())
            max_weight=np.array([max_value]*self.cluster_num)
            minus_weight=np.array([max_minus]*self.cluster_num)
            weight_pro=np.array([1]*self.cluster_num)-(max_weight-weight_embed)/minus_weight
            cluster_vector.append(weight_pro)
        return np.array(cluster_vector)

    def get_embedding(self,token_list,option="query_embedding",dim=300):
        if(option=="query_embedding"):
            return self.clusterpro_vector(token_list,dim=dim)
        else:
            return self.BOSW_vector(token_list,dim=dim)

    def get_vector(self,word,dim=300):
        try:
            return self.model[word]
        except:
            return [0]*dim
