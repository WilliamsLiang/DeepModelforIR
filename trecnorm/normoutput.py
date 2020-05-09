import re
from evalfuntion.retrivalEval import NDCG


class normTrec():
    """
    normTrec 标准化输出工具，输出为特定文本
    """
    def __init__(self,isList=False):
        """
        :param isList:所输入的对象是否为列表
        """
        self.isList=isList

    def normData(self,result,query_id="Fault",modelname="Fault",subtopic="0"):
        """
        :param result:支持两种输入格式 列表[[docid,相似度]] 字典{"docid":相似度}
        :param query_id:输出的query_id
        :param modelname:输出的模型名称
        :param subtopic:输出的query子主题
        :return:返回标准化后的文本列表
        """
        if(not self.isList):
            tmp_list=[[key,result[key]] for key in result.keys()]
            sort_list=sorted(tmp_list,key=lambda x:x[1],reverse=True)
        else:
            sort_list=self.covertList(result)
        tmp_result=[]
        i=0
        for data in sort_list:
            i=i+1
            tmp_result.append("\t".join([query_id,subtopic,data[0],str(i),str(data[1]),modelname]))
        return tmp_result

    def covertList(self,result):
        sort_list = sorted(result, key=lambda x: x[1], reverse=True)
        return sort_list


class trecEval():
    """
    标准化评估工具，可能与官方工具存在不同
    """
    def __init__(self):
        """
        默认使用NDCG算法评价
        simDict存储相关度打分
        resultDict处理成格式：范例{ "docid":{"sim": relevant(人工打分的相似度),"rank":int(模型排序结果)}}
        """
        self.evalMethod=NDCG()
        self.simDict={}
        self.resultDict={}
        self.value_re=re.compile("[a-zA-Z]")
        pass

    def load_trecFile(self,resultFile="",similarFile="",split_tag=" "):
        """
        该函数时类加载文件的基础函数，符合TREC的标准文档时，推荐使用该函数
        :param resultFile: TREC 实验模型的标准化输出结果
        :param similarFile: TREC 所提供的人工相似度文件
        :param split_tag: 文件列分隔符
        :return:
        """
        self.loadSimilar(similarFile,split_tag=split_tag)
        self.loadResult(resultFile,split_tag=split_tag)
        return self

    def loadResult(self,resultFile="",qIndex=0,subqindex=-1,dIndex=2,rankIndex=3,split_tag=" "):
        """
        :param resultFile:实验模型的标准化输出结果
        :param qIndex: 输出结果的 query 索引
        :param subqindex:输出结果的 subtopic 索引，不存在则为-1
        :param dIndex:输出结果的 docid 索引
        :param rankIndex:输出结果的 rank 索引
        :param split_tag:分隔符
        :return:self
        """
        f = open(resultFile, "rt", encoding="utf-8")
        maxindex = max([qIndex, dIndex,rankIndex])
        line = f.readline()
        while (line):
            datas = line.replace("\r", "").replace("\n", "").split(split_tag)
            if (len(datas) < maxindex):
                line = f.readline()
                continue
            if(subqindex!=-1):
                q_id = datas[qIndex]+"_"+datas[subqindex]
            else:
                q_id = datas[qIndex]
            doc_id = datas[dIndex]
            rank=datas[rankIndex]
            self.resultDict[q_id]=self.resultDict.get(q_id,{})
            self.resultDict[q_id][doc_id]={"rank":int(rank),"sim":self.simDict.get(q_id,{}).get(doc_id,{}).get("sim",0)}
            line = f.readline()
        f.close()
        for key in self.resultDict.keys():
            tmp_list=[[doc_id,self.resultDict[key][doc_id]["rank"]] for doc_id in self.resultDict[key].keys()]
            sort_list=sorted(tmp_list,key=lambda x:x[1],reverse=False)
            for i in range(len(sort_list)):
                self.resultDict[key][sort_list[i][0]]["rank"]=(i+1)
        return self

    def loadSimilar(self,similarFile="",qIndex=0,subqindex=-1,dIndex=2,simIndex=3,split_tag=" "):
        """
        :param similarFile:人工相似度标签
        :param qIndex: 相似度文档的 query 索引
        :param subqindex:相似度文档的 subtopic 索引，不存在则为-1
        :param dIndex:相似度文档的 docid 索引
        :param simIndex:相似度文档的 相似度 索引
        :param split_tag:分隔符
        :return:self
        """
        f=open(similarFile,"rt",encoding="utf-8")
        maxindex=max([qIndex,dIndex,simIndex])
        line=f.readline()
        while(line):
            datas=line.replace("\r","").replace("\n","").split(split_tag)
            if(len(datas)<maxindex):
                line=f.readline()
                continue
            if (subqindex != -1):
                q_id = datas[qIndex] + "_" + datas[subqindex]
            else:
                q_id = datas[qIndex]
            doc_id=datas[dIndex]
            sim=self.get_value(datas[simIndex])
            self.simDict[q_id]=self.simDict.get(q_id,{})
            self.simDict[q_id][doc_id]={"sim":float(sim)}
            line=f.readline()
        f.close()
        for key in self.simDict.keys():
            tmp_list=[[doc_id,self.simDict[key][doc_id]["sim"]] for doc_id in self.simDict[key].keys()]
            sort_list=sorted(tmp_list,key=lambda x:x[1],reverse=True)
            for i in range(len(sort_list)):
                self.simDict[key][sort_list[i][0]]["rank"]=(i+1)
        return self

    def get_value(self,value):
        return self.value_re.sub("",value)

    def get_avgNDCG(self,n):
        """
        该函数计算所有BASELIN文件的NDCG值，没有给人工相似度的为0
        :param n:NDCG指标中的N值
        :return:模型输出的平均值
        """
        res_list=[]
        for key in self.resultDict.keys():
            if(not self.simDict.get(key,{})):
                continue
            self.evalMethod.modifybaselines_general(self.simDict[key])
            self.evalMethod.modifyrank_general(self.resultDict[key])
            res_list.append(self.evalMethod.getDcg(n))
            #print(key + ":" +str(self.evalMethod.getDcg(n)))
        return float(sum(res_list))/len(res_list)

    def get_avgNDCGfilter(self,n):
        """
        该函数只计算人工打分文件中的docid的整体表现情况，不考虑全部的BASELINE文件
        :param n:NDCG指标中的N值
        :return:模型输出的平均值
        """
        res_list = []
        resultdict=self.resultDict.copy()
        for key in self.resultDict.keys():
            if (not self.simDict.get(key, {})):
                continue
            tmp_list=[[docid,resultdict[key][docid]] for docid in self.simDict[key].keys() if(resultdict[key].get(docid,None))]
            tmp_list=sorted(tmp_list,key=lambda x:x[1]["rank"],reverse=False)
            tmp_dict={}
            for i in range(len(tmp_list)):
                tmp_list[i][1]["rank"]=(i+1)
                tmp_dict[tmp_list[i][0]]=tmp_list[i][1]
            self.evalMethod.modifybaselines_general(self.simDict[key])
            self.evalMethod.modifyrank_general(tmp_dict)
            res_list.append(self.evalMethod.getDcg(n))
            # print(key + ":" +str(self.evalMethod.getDcg(n)))
        return float(sum(res_list)) / len(res_list)


if __name__=="__main__":
    similarFile="D:/任务_待解决/比赛_NTCIR/WWW-3数据/www2e.qrels"
    resultFile="D:/任务_待解决/比赛_NTCIR/比赛结果/英文任务/ltr_rep_1.txt"
    evaltool=trecEval()
    evaltool.loadSimilar(similarFile, qIndex=0, dIndex=1, simIndex=2, split_tag=" ")
    evaltool.loadResult(resultFile,qIndex=0,dIndex=2,rankIndex=3,split_tag=" ")
    print(evaltool.get_avgNDCGfilter(1))
    print(evaltool.get_avgNDCGfilter(5))
    print(evaltool.get_avgNDCGfilter(10))
    print(evaltool.get_avgNDCGfilter(20))
    print("--------------------------")

    similarFile = "D:/任务_待解决/比赛_NTCIR/WWW-3数据/www2e.qrels"
    resultFile = "D:/任务_待解决/比赛_NTCIR/比赛结果/英文任务/ltr_rep_1.txt"
    evaltool = trecEval()
    evaltool.loadSimilar(similarFile, qIndex=0, dIndex=1, simIndex=2, split_tag=" ")
    evaltool.loadResult(resultFile, qIndex=0, dIndex=2, rankIndex=3, split_tag=" ")
    print(evaltool.get_avgNDCG(1))
    print(evaltool.get_avgNDCG(5))
    print(evaltool.get_avgNDCG(10))
    print(evaltool.get_avgNDCG(20))



