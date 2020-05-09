import pyltr
import os
import re

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


docid_re = re.compile(r"docid {0,1}= {0,1}([a-zA-Z0-9\-]*)")

class normTrec():
    def __init__(self,isList=False):
        self.isList=isList

    def normData(self,result,query_id="Fault",modelname="Fault",subtopic="0"):
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

def convertResult(y,qids,dids):
    tmp_dict={}
    for i in range(len(qids)):
        value=y[i]
        qid=qids[i]
        result=docid_re.findall(dids[i])
        if(result):
            doc_id=result[0]
        else:
            doc_id=""
        tmp_dict[qid]=tmp_dict.get(qid,{})
        tmp_dict[qid][doc_id]=value
    return tmp_dict


train_path = r"/home/user/ntcir_match/base_run/ltr_train_all.txt"
test_path = r'/home/user/ntcir_match/base_run/ltr_test.txt'
with open(train_path) as trainfile, \
        open(train_path) as valifile, \
        open(test_path) as evalfile:
    print('---start read.')
    TX, Ty, Tqids, Tdocid = pyltr.data.letor.read_dataset(trainfile,has_targets=True)
    print('---end read.')
    # VX, Vy, Vqids, _ = pyltr.data.letor.read_dataset(valifile)
    EX, Ey, Eqids, Edocid = pyltr.data.letor.read_dataset(evalfile,has_targets=True)
    trec_out=normTrec()
    metric_ap_10 = pyltr.metrics.AP(k=10)
    #metric_10 = pyltr.metrics.NDCG(k=10)
    #metric_5 = pyltr.metrics.NDCG(k=5)
    #metric_1 = pyltr.metrics.NDCG(k=1)

    # Only needed if you want to perform validation (early stopping & trimming)
    monitor = pyltr.models.monitors.ValidationMonitor(
        TX, Ty, Tqids, metric=metric_ap_10, stop_after=200)

    model = pyltr.models.LambdaMART(
        metric=metric_ap_10,
        n_estimators=5000,
        learning_rate=0.02,
        max_features=0.5,
        query_subsample=0.5,
        max_leaf_nodes=10,
        min_samples_leaf=64,
        verbose=1,
    )
    print('---start fit.')
    model.fit(TX, Ty, Tqids, monitor=monitor)
    print('---end fit.')
    Epred = model.predict(EX)
    # print(Epred)
    # print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
    print('Our model AP(10):', metric_ap_10.calc_mean(Eqids, Ey, Epred))
    #print('Our model NDCG(5):', metric_5.calc_mean(Eqids, Ey, Epred))
    #print('Our model NDCG(10):', metric_10.calc_mean(Eqids, Ey, Epred))
    #print('Our model NDCG(20):', metric_20.calc_mean(Eqids, Ey, Epred))
    w = open("/home/user/ntcir_match/base_run/letor_test/letor_result.txt", "wt", encoding="utf-8")
    result_dict=convertResult(Epred,Eqids,Edocid)
    for key in result_dict.keys():
        outlines = "\n".join(trec_out.normData(result_dict[key], query_id=key, modelname="ltr"))
        w.write(outlines + "\n")
    w.close()
