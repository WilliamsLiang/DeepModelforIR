#coding=utf-8


class prfeval():
    def __init__(self,settype="class",mainseq={"BE-ft":[["B","E"],["BE"]]},othseq=["S"]):
        self.setype=settype
        if(settype=="seq"):
            self.mainseq=mainseq
            self.othseq=othseq
            self.mainseqindex={}
            for key in self.mainseq.keys():
                for label_list in self.mainseq[key]:
                    start_label=label_list[0]
                    end_label=label_list[-1]
                    self.mainseqindex[start_label]=[key,end_label]
        pass

    def evalset(self,resultset,orindex=-2,preindex=-1):
        if(self.setype=="class"):
            return self.evalset_class(resultset,orindex,preindex)
        elif(self.setype=="seq"):
            return self.evalset_seq(resultset, orindex, preindex)
        pass

    def evalset_class(self,resultset,orindex=-2,preindex=-1):
        '''
        :param resultset: 预测结果集 范例：[文本内容,原始标签，预测标签]
        :param orindex: 原始标签索引
        :param preindex: 预测标签索引
        :return: 返回一个计算结果dict
        '''
        label_p_dict={}
        label_r_dict={}
        for result in resultset:
            orilabel=result[orindex]
            prelabel=result[preindex]
            label_p_dict[prelabel] = label_p_dict.get(prelabel,[0,0])
            label_r_dict[orilabel] = label_r_dict.get(orilabel, [0, 0])
            label_p_dict[orilabel] = label_p_dict.get(orilabel, [0, 0])
            label_r_dict[prelabel] = label_r_dict.get(prelabel, [0, 0])
            if(orilabel==prelabel):
                label_p_dict[prelabel][0]=label_p_dict[prelabel][0]+1
                label_r_dict[orilabel][0]=label_r_dict[orilabel][0]+1
            else:
                label_p_dict[prelabel][1] = label_p_dict[prelabel][1] + 1
                label_r_dict[orilabel][1] = label_r_dict[orilabel][1] + 1
        label_dict={}
        key_set=set(list(label_r_dict.keys())+list(label_p_dict.keys()))
        for key in key_set:
            if((label_r_dict[key][0]+label_r_dict[key][1])!=0):
                r_value=label_r_dict[key][0]/(label_r_dict[key][0]+label_r_dict[key][1])
            else:
                r_value=-1
            if ((label_p_dict[key][0] + label_p_dict[key][1]) != 0):
                p_value = label_p_dict[key][0] / (label_p_dict[key][0] + label_p_dict[key][1])
            else:
                p_value = -1
            if((p_value != -1 or p_value!=0) and (r_value != -1 or r_value!=0)):
                f_value=2*p_value*r_value/(p_value+r_value)
            else:
                f_value=-1
            label_dict[key]={"p":p_value,"r":r_value,"f":f_value}
        return label_dict

    def evalset_seq(self,resultset,orindex=-2,preindex=-1):
        '''
        :param resultset: 预测结果集 范例：[word,原始标签，预测标签]
        :param orindex: 原始标签索引
        :param preindex: 预测标签索引
        :return: 返回一个计算结果dict
        这个函数待重构，无法满足大量数据计算。原因：所有句子都要合并成一个句子计算
        还有一点该函数代码量太多，也需要重构，这个等待思考
        '''
        label_p_dict = {}
        label_r_dict = {}
        pre_end_label_tag=""
        ori_end_label_tag=""
        pre_key=""
        ori_key=""
        p_seq_flag = False
        r_seq_flag = False
        p_flag=False
        r_flag=False
        for result in resultset:
            pre_label=result[preindex]
            ori_label=result[orindex]
            same_flag=(pre_label==ori_label)
            if(not p_seq_flag):
                pre_key , pre_end_label_tag=self.mainseqindex.get(pre_label,["",""])
                if(not pre_key):
                    pre_key=pre_label
                else:
                    p_seq_flag=True
                    p_flag=True
            if (not r_seq_flag):
                ori_key , ori_end_label_tag = self.mainseqindex.get(ori_label, ["", ""])
                if (not ori_key):
                    ori_key = ori_label
                else:
                    r_seq_flag = True
                    r_flag = True
            label_p_dict[pre_key] = label_p_dict.get(pre_key, [0, 0])
            label_r_dict[ori_key] = label_r_dict.get(ori_key, [0, 0])
            label_p_dict[ori_key] = label_p_dict.get(ori_key, [0, 0])
            label_r_dict[pre_key] = label_r_dict.get(pre_key, [0, 0])
            if((((pre_label in self.othseq) and (not p_seq_flag)) or (p_seq_flag and p_flag and (pre_label==pre_end_label_tag))) and same_flag ):
                label_p_dict[pre_key][0] = label_p_dict[pre_key][0] + 1
                p_flag = False
                p_seq_flag = False
                pre_end_label_tag = ""
            elif((pre_label in self.othseq) or (p_seq_flag and (pre_label==pre_end_label_tag))):
                label_p_dict[pre_key][1] = label_p_dict[pre_key][1] + 1
                r_flag = False
                p_seq_flag = False
                pre_end_label_tag = ""
            if ((((ori_label in self.othseq) and (not r_seq_flag)) or (r_seq_flag and r_flag and (ori_label == ori_end_label_tag))) and same_flag):
                label_r_dict[ori_key][0] = label_r_dict[ori_key][0] + 1
                r_flag = False
                r_seq_flag = False
                ori_end_label_tag = ""
            elif ((ori_label in self.othseq) or (r_seq_flag and (ori_label == ori_end_label_tag))):
                label_r_dict[ori_key][1] = label_r_dict[ori_key][1] + 1
                r_flag = False
                r_seq_flag = False
                ori_end_label_tag = ""
            if((pre_label not in self.othseq) and p_seq_flag and (not same_flag)):
                p_flag = False
            if ((ori_label not in self.othseq) and r_seq_flag and (not same_flag)):
                r_flag = False
        label_dict = {}
        key_set = set(list(label_r_dict.keys()) + list(label_p_dict.keys()))
        for key in key_set:
            if ((label_r_dict[key][0] + label_r_dict[key][1]) != 0):
                r_value = label_r_dict[key][0] / (label_r_dict[key][0] + label_r_dict[key][1])
            else:
                r_value = -1
            if ((label_p_dict[key][0] + label_p_dict[key][1]) != 0):
                p_value = label_p_dict[key][0] / (label_p_dict[key][0] + label_p_dict[key][1])
            else:
                p_value = -1
            if ((p_value != -1 or p_value!=0) and (r_value != -1 or r_value!=0)):
                f_value = 2 * p_value * r_value / (p_value + r_value)
            else:
                f_value = -1
            label_dict[key] = {"p": p_value, "r": r_value, "f": f_value}
        return label_dict
        pass
