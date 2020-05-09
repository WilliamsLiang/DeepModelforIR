import os

from util.bigcount import BigCount_multi


def main():
    file_path="E:/CLUBWEB_experiment/ntcir_www2/result_html_eng/"
    file_list=os.listdir(file_path)
    model=BigCount_multi(query_file="C:/Users/sfe_williamsL/Desktop/任务_待解决/比赛_NTCIR/WWW-3数据/www2www3topics-E.xml")
    for file in file_list:
        print(file)
        f=open(file_path+file,"rb")
        data=f.read().decode("utf-8","ignore").split("\r\n\r\n")
        html_code="\n".join(data[1:])
        print(len(html_code))
        model.count_single(html_code)
        f.close()

    key_list=["title","body","a","html"]
    for key in key_list:
        w=open("E:/CLUBWEB_experiment/ntcir_www2/"+key+"_idf.txt","wt",encoding="utf-8")
        tmp_dict=model.get_dict(key)
        for key in tmp_dict.keys():
            w.write("\t".join([key]+[str(value) for value in tmp_dict[key]])+"\n")
        w.close()



if __name__=="__main__":
    main()