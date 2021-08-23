import sys

sys.path.append(r'/home/user/ntcir_match/lzdeep/NTCIR_project/')

from model.drmm import DRMM,DrmmDataloader
from util.basefunction import load_evalfile,shuffle_rank,load_baseline
from trecnorm.normoutput import normTrec

if __name__=="__main__":
    query_dict = {}
    doctext_dict = {}
    max_doc = 1000
    baseline_path = "/home/user/ntcir_match/en_task/baselineEng.txt"
    htmlfile_path = "/home/user/ntcir_match/en_task/result_html_eng/"
    queryfile="/home/user/ntcir_match/en_task/www2www3topics-E.xml"
    evalfile="/home/user/ntcir_match/en_task/www2e.qrels"
    model_path = "/home/user/ntcir_match/lzdeep/modelpath/DRMM/"
    idf_file="/home/user/ntcir_match/en_task/html_idf.txt"
    embedfile="/home/user/ntcir_match/lzdeep/word2vec/word2vec.model"
    data_loader = DrmmDataloader(bin=0.5,
                 querynum=5,
                 idf_file=idf_file,
                 embedfile=embedfile,
                 queryfile=queryfile,
                 data_path = htmlfile_path,
                 wc_index=2)

    eval_dict = load_evalfile(evalfile)
    
    train_data = shuffle_rank(eval_dict)
    print(len(train_data))
    model = DRMM(train_flag=True,model_path = model_path,batch_size=32)
    print("model is being loaded....")
    model.train(train_data,data_loader)
    print("model has been loaded....")
    testdata =load_baseline(baseline_path)
    result_dict = {}
    for key in testdata.keys():
        test_data = []
        for docid in testdata[key]:
            test_data.append([key,docid])
            if(len(test_data)>=max_doc):
                
                result = model.predict(test_data,data_loader)
                for q in result.keys():
                    result_dict[q]=result_dict.get(q,{})
                    for doc in result[q].keys():
                        result_dict[q][doc]=result[q][doc]
                data_loader.cache_clear()
                print("1000 file has been computed.....")
        result = model.predict(test_data,data_loader)
        for q in result.keys():
            result_dict[q]=result_dict.get(q,{})
            for doc in result[q].keys():
                result_dict[q][doc]=result[q][doc]
        data_loader.cache_clear()
        print(key+" query has been predicted....")
    
    trec_out=normTrec()
    w = open("/home/user/ntcir_match/lzdeep/drmm/drmm_top1000.txt", "wt", encoding="utf-8")
    for key in result_dict.keys():
        outlines = "\n".join(trec_out.normData(result_dict[key], query_id=key, modelname="DRMM",split_tag=" "))
        w.write(outlines + "\n")
    w.close()