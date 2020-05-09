#coding=utf-8
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
        '''
        :param all_dict:范例{ "docid":{"sim": relevant(人工打分的相似度),"rank":int(模型排序结果)}}
        :return:返回该对象
        '''
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
        """
        :param valuelist:数据集：[[simvalue(模型排序结果),relindex(人工打分结果结果)]]
        :param simindex: 索引调整
        :param relindex: 索引调整
        :return:
        """
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