import gensim

import re
import os
import nltk.stem.porter as pt

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec

pt_stemmer = pt.PorterStemmer()
pat_letter = re.compile(r'[a-zA-Z\-\'\.]+')
oth_re=re.compile(r"<!\[.*?>")
cachedStopWords = set(stopwords.words("english"))

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

class HtmlParseBybs:
    def __init__(self):
        self.soupstring=None
        pass

    def parse(self,html):
        html = oth_re.sub("", html)
        self.soupstring=BeautifulSoup(html, "html.parser")
        [s.extract() for s in self.soupstring("script")]
        return self

    def get_text(self,tag):
        """
        :param tag:HTML 标签
        :return:解析标签之后的文本内容
        """
        tag_list=[_.get_text() for _ in self.soupstring.select(tag)]
        return tag_list

    def parse_get(self,html,tag):
        """
        :param html:带解析的HTML 字符串
        :param tag: 获取的标签
        :return:  解析标签之后的文本内容
        """
        html=oth_re.sub("",html)
        self.soupstring = BeautifulSoup(html, "html.parser")
        [s.extract() for s in self.soupstring("script")]
        tag_list = [_.get_text() for _ in self.soupstring.select(tag)]
        return tag_list

class HtmlInfo:
    def __init__(self,html):
        self.bsmodel=HtmlParseBybs().parse(html)

    def get_tokenlist(self,tag):
        tmp_list = self.bsmodel.get_text(tag)
        result= [pt_stemmer.stem(w.lower()) for tmp in tmp_list for w in pat_letter.findall(tmp) if(w not in cachedStopWords)]
        return result

    def get_tokendict(self,tag):
        tmp_list = self.bsmodel.get_text(tag)
        tmp_dict = {"doc_lenth":0}
        for tmp in tmp_list:
            tag_list = pat_letter.findall(tmp)
            for token in tag_list:
                if (token in cachedStopWords):
                    continue
                token=pt_stemmer.stem(token.lower())
                tmp_dict[token]=tmp_dict.get(token,0)+1
                tmp_dict["doc_lenth"]=tmp_dict["doc_lenth"]+1
        return tmp_dict


filepath="/home/user/ntcir_match/en_task/result_html_eng/"
url_re = re.compile("X-INKT-URI:(.*?)\n")
file_list=os.listdir(filepath)
i=0
sentences=[]
model= Word2Vec(size=128, min_count=2, window=5)
for file in file_list:
    f=open(filepath+file,"rb")
    data=f.read().decode("utf-8","ignore")
    f.close()
    html_code="\n".join(data.split("\r\n\r\n")[1:])
    baseinfo=data.split("\r\n\r\n")[0]
    url_info = "/".join(url_re.findall(baseinfo)).split("/")
    maininfo=HtmlInfo(html_code)
    result_dict={}
    token_list = maininfo.get_tokenlist("html")
    all_token=url_info+token_list
    i=i+1
    sentences.append(all_token)
    if(i%1000==0 and i!=0):
        model.build_vocab(sentences=sentences,update=True)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        i=0
        sentnces=[]
model.build_vocab(sentences=sentences,update=True)
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
model.save("word2vec.model")