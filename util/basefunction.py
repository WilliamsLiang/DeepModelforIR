import re
import random

value_re=re.compile("[a-zA-Z]")

def filter_tags(htmlstr):
    #先过滤CDATA
    re_cdata=re.compile('//<![CDATA[[^>]*//]]>',re.I) #匹配CDATA
    re_script=re.compile('<s*script[^>]*>[^<]*<s*/s*scripts*>',re.I)#Script
    re_style=re.compile('<s*style[^>]*>[^<]*<s*/s*styles*>',re.I)#style
    re_br=re.compile('<brs*?/?>')#处理换行
    re_h=re.compile('</?[^>]+?>')#HTML标签
    re_comment=re.compile('<!--[^>]*-->')#HTML注释
    s=re_cdata.sub('',htmlstr)#去掉CDATA
    s=re_script.sub('',s) #去掉SCRIPT
    s=re_style.sub('',s)#去掉style
    s=re_br.sub('n',s)#将br转换为换行
    s=re_h.sub('',s) #去掉HTML 标签
    s=re_comment.sub('',s)#去掉HTML注释
    #去掉多余的空行
    blank_line=re.compile('n+')
    s=blank_line.sub('n',s)
    s=replaceCharEntity(s)#替换实体
    return s

def shuffle_pos(eval_dict,neg_docs = 4):
    data_list = []
    for key in eval_dict.keys():
        pos_list = []
        neg_list = []
        for docid in eval_dict[key].keys():
            if(eval_dict[key][docid]>0):
                pos_list.append(docid)
            else:
                neg_list.append(docid)
        for posid in pos_list:
            if(len(neg_list)<neg_docs):
                continue
            neglist = random.sample(neg_list, neg_docs)
            data_list.append([key,[posid]+neglist])
    return data_list

def shuffle_rank(eval_dict,sample = 1):
    data_list = []
    for key in eval_dict.keys():
        docid_list = eval_dict[key].keys()
        for docid in docid_list:
            posid = docid
            posvalue = eval_dict[key][docid]
            negid_list = []
            for negid in docid_list:
                if(eval_dict[key][negid]<posvalue):
                    negid_list.append(negid)
            if(len(negid_list)<sample):
                for negid in negid_list:
                    data_list.append([key,[posid,negid]])
            else:
                for negid in random.sample(negid_list, sample):
                    data_list.append([key,[posid,negid]])
    return data_list

def shuffle_rel(eval_dict):
    data_list = []
    y_label = []
    q_id = list(eval_dict.keys())
    for key in eval_dict.keys():
        docid_list = eval_dict[key].keys()
        for docid in docid_list:
            value = eval_dict[key][docid]
            data_list.append([key,docid])
            y_label.append(value)
    return data_list,y_label,q_id

def get_value(value):
    return value_re.sub("",value)

def load_baseline(basefile, qIndex=0, subqindex=-1, dIndex=2, split_tag=" "):
    testdata={}
    f = open(basefile, "rt", encoding="utf-8")
    for line in f.readlines():
        datas = line.split(split_tag)
        if (len(datas) < max([qIndex, subqindex, dIndex])):
            continue
        q_id = datas[qIndex]
        if (subqindex != -1):
            q_id = q_id + "_" + datas[subqindex]
        doc_id = datas[dIndex]
        testdata[q_id]=testdata.get(q_id,[])
        testdata[q_id].append(doc_id)
    f.close()
    return testdata

def load_evalfile(similarFile, qIndex=0, subqindex=-1, dIndex=1, simIndex=2, split_tag=" "):
    """
    :param similarFile:人工相似度标签
    :param qIndex: 相似度文档的 query 索引
    :param subqindex: 相似度文档的 subtopic 索引，不存在则为-1
    :param dIndex: 相似度文档的 docid 索引
    :param simIndex: 相似度文档的 相似度 索引
    :param split_tag: 分隔符
    :return:self
    """
    eval_dict = {}
    f = open(similarFile, "rt", encoding="utf-8")
    maxindex = max([qIndex, dIndex, simIndex])
    line = f.readline()
    while (line):
        datas = line.replace("\r", "").replace("\n", "").split(split_tag)
        if (len(datas) < maxindex):
            line = f.readline()
            continue
        if (subqindex != -1):
            q_id = datas[qIndex] + "_" + datas[subqindex]
        else:
            q_id = datas[qIndex]
        if(int(q_id)>80):
            line = f.readline()
            continue
        doc_id = datas[dIndex]
        sim = get_value(datas[simIndex])
        eval_dict[q_id] = eval_dict.get(q_id, {})
        eval_dict[q_id][doc_id] = int(sim)
        line = f.readline()
    f.close()
    return eval_dict

def replaceCharEntity(htmlstr):
    CHAR_ENTITIES = {'nbsp': ' ', '160': ' ',
                     'lt': '<', '60': '<',
                     'gt': '>', '62': '>',
                     'amp': '&', '38': '&',
                     'quot': '"', '34': '"', '8217':"'"}

    re_charEntity = re.compile(r'&#?(?P<name>\w+);')
    sz = re_charEntity.search(htmlstr)
    while sz:
        key = sz.group('name')  # 去除&;后entity,如>为gt
        try:
            htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
        except KeyError:
            # 以空串代替
            htmlstr = re_charEntity.sub('', htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
    return htmlstr


def repalce(s, re_exp, repl_string):
    return re_exp.sub(repl_string, s)