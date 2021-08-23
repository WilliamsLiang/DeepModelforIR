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


    def miDcg(self,n):
        repl = 0.0
        for key in self.data.keys():
            index = self.data[key]["rank"]
            if (index > n):
                continue
            repl = repl + (math.pow(2,self.data[key]["sim"])-1) / math.log(self.data[key]["rank"]+1, 2)
        return repl

    def addDcg(self,n):
        repl = 0.0
        for key in self.data.keys():
            index = self.data[key]["rank"]
            if (index > n):
                continue
            repl = repl + self.data[key]["sim"]
        return repl

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



class ERR:
    def __init__(self):
        self.data={}
        pass

    def modifyrank(self,all_dict):
        """
        :param all_dict:数据规范{"docid":{"sim":人工相似度,"rank":排序位置}}
        :return:
        """
        self.data=all_dict
        return self

    def covertList(self,valuelist,simindex=0,relindex=1):
        """
        :param valuelist:数据规范 [[模型相似度，人工相似度]]
        :param simindex:模型相似索引
        :param relindex:人工相似度索引
        :return:
        """
        baselist=sorted(valuelist,key=lambda x:x[simindex],reverse=True)
        for i in range(len(baselist)):
            self.data[i]={"sim":baselist[i][relindex],"rank":i+1}
        return self

    def getERR(self,n,max_rel=2):
        """
        :param n: ERR指标中的N值，只返回前多少结果
        :return:
        """
        value=0
        pre_rel=1
        rank_list=sorted([[self.data[key]["rank"],self.data[key]["sim"]] for key in self.data.keys()],key=lambda x:x[0])
        #max_rel = max([self.data[key]["sim"] for key in self.data.keys()])
        for i in range(min(len(rank_list),n)):
            r=i+1
            r_rel=float((math.pow(2,rank_list[i][1])-1))/math.pow(2,max_rel)
            tmp_value=(1.0/r)*pre_rel*r_rel
            pre_rel=pre_rel*(1-r_rel)
            value=value+tmp_value
        return value

class nERR:
    def __init__(self):
        self.baseErr=ERR()
        self.rankErr=ERR()

    def modifybaselines_general(self,all_dict):
        """
        :param all_dict:数据规范{"docid":{"sim":人工相似度,"rank":理想排序位置}}
        :return:
        """
        self.baseErr.modifyrank(all_dict)
        return self

    def modifyrank_general(self,all_dict):
        """
        :param all_dict:数据规范{"docid":{"sim":人工相似度,"rank":排序位置}}
        :return:
        """
        self.rankErr.modifyrank(all_dict)
        return self

    def covertList(self,valuelist,simindex=0,relindex=1):
        """
        :param valuelist:数据规范 [[模型相似度，人工相似度]]
        :param simindex:模型相似索引
        :param relindex:人工相似度索引
        :return:
        """
        baselist=[ [value[relindex],value[relindex]] for value in valuelist]
        self.baseErr.covertList(baselist,simindex,relindex)
        self.rankErr.covertList(valuelist, simindex, relindex)
        return self

    def getNERR(self,n,max_rel=2):
        """
        :param n: NDCG指标中的N值，只返回前多少结果
        :param dcgtype: DCG的计算公式分为:miDCG 和 baseDCG
        :return:
        """
        return self.rankErr.getERR(n,max_rel=max_rel)/self.baseErr.getERR(n,max_rel=max_rel)


class Precesion:
    def __init__(self):
        self.data={}
        pass

    def modifyrank(self,all_dict):
        """
        :param all_dict:数据规范{"docid":{"sim":人工相似度,"rank":排序位置}}
        :return:
        """
        self.data=all_dict
        return self

    def covertList(self,valuelist,simindex=0,relindex=1):
        """
        :param valuelist:数据规范 [[模型相似度，人工相似度]]
        :param simindex:模型相似索引
        :param relindex:人工相似度索引
        :return:
        """
        baselist=sorted(valuelist,key=lambda x:x[simindex],reverse=True)
        for i in range(len(baselist)):
            self.data[i]={"sim":baselist[i][1],"rank":i+1}
        return self

    def getP(self,n):
        """
        :param n: ERR指标中的N值，只返回前多少结果
        :return:
        """
        max_value = 0.0
        rank_list = sorted([[self.data[key]["rank"], self.data[key]["sim"]] for key in self.data.keys()],
                           key=lambda x: x[0])
        for i in range(min(len(rank_list), n)):
            if(rank_list[i][1]>0):
                max_value+=1
        return max_value/n

class Qmeasure:
    def __init__(self,beta=1.0):
        self.baseDcg = DCG()
        self.rankDcg = DCG()
        self.data={}
        self.beta=beta

    def modifybaselines_general(self, all_dict):
        """
        :param all_dict:数据规范{"docid":{"sim":人工相似度,"rank":理想排序位置}}
        :return:
        """
        self.baseDcg.modify_rank(all_dict)
        return self

    def modifyrank_general(self, all_dict):
        """
        :param all_dict:数据规范{"docid":{"sim":人工相似度,"rank":排序位置}}
        :return:
        """
        self.rankDcg.modify_rank(all_dict)
        self.data=all_dict
        return self

    def covertList(self, valuelist, simindex=0, relindex=1):
        """
        :param valuelist:数据规范 [[模型相似度，人工相似度]]
        :param simindex:模型相似索引
        :param relindex:人工相似度索引
        :return:
        """
        baselist = [[value[relindex], value[relindex]] for value in valuelist]
        self.baseDcg.convertList(baselist, simindex, relindex)
        self.rankDcg.convertList(valuelist, simindex, relindex)
        ranklist = sorted(valuelist, key=lambda x: x[simindex], reverse=True)
        for i in range(len(ranklist)):
            self.data[i] = {"sim": ranklist[i][1], "rank": i + 1}
        return self

    def getQm(self, n):
        """
        :param n: NDCG指标中的N值，只返回前多少结果
        :param dcgtype: DCG的计算公式分为:miDCG 和 baseDCG
        :return:
        """
        value=0.0
        rank_list = sorted([[self.data[key]["rank"], self.data[key]["sim"]] for key in self.data.keys()],
                           key=lambda x: x[0])
        r_number = len([_ for _ in rank_list if(_[1]>0)])
        pre_value=0.0
        for i in range(min(len(rank_list),n)):
            i_value=0
            rank = i+1
            if(rank_list[i][1]>0):
                i_value=1
            pre_value=pre_value+i_value
            value=value+i_value*((pre_value+self.beta*self.rankDcg.addDcg(rank))/(i+1+self.beta*self.baseDcg.addDcg(rank)))
        return value/min(n,r_number)