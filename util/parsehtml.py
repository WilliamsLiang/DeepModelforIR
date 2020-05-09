import re
import nltk.stem.porter as pt

from bs4 import BeautifulSoup
from util.basefunction import filter_tags
from nltk.corpus import stopwords

pt_stemmer = pt.PorterStemmer()
pat_letter = re.compile(r'[a-zA-Z\-\'\.]+')
oth_re=re.compile(r"<!\[.*?>")
cachedStopWords = set(stopwords.words("english"))

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

class HtmlParseByre:
    def __init__(self):
        self.htmlstring=None
        self.scriptre=re.compile(r"<script.*?</script>")
        pass

    def parse(self,html):
        self.htmlstring=self.scriptre.sub("",html)
        return self

    def get_text(self,tag):
        """
        :param tag:HTML 标签
        :return:解析标签之后的文本内容
        """
        self.tagre=re.compile(r"<"+tag+"[\S\s]*?</"+tag+">")
        tag_list=[filter_tags(_) for _ in self.tagre.findall(self.htmlstring)]
        return tag_list

    def parse_get(self,html,tag):
        """
        :param html:带解析的HTML 字符串
        :param tag: 获取的标签
        :return:  解析标签之后的文本内容
        """
        self.htmlstring=self.scriptre.sub("",html)
        self.tagre = re.compile(r"<" + tag + "[\S\s]*?</" + tag + ">")
        tag_list = [filter_tags(_) for _ in self.tagre.findall(self.htmlstring)]
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



if __name__=="__main__":
    f=open("E:/CLUBWEB_experiment/ntcir_www2/result_html_eng/clueweb12-0806wb-66-10978.html","rb")
    pt_stemmer = pt.PorterStemmer()

    html_code=" ".join(f.read().decode("utf-8","ignore").split("\r\n\r\n")[1:]).replace("\r\n"," ").replace("\t","")
    f.close()
    print(html_code)

    a_list=HtmlParseBybs().parse_get(html_code,"body")
    for a in a_list:
        print(a)
        tag_list=pat_letter.findall(a)
        #tag =[w.lower() for w in word_tokenize(a)]
        tag = [w.lower() for w in tag_list]
        print(tag)
        word_list=[pt_stemmer.stem(w) for w in tag]
        print(word_list)
    '''
    a_list = HtmlParseByre().parse_get(html_code, "body")
    for a in a_list:
        print(a)
        tag_list = pat_letter.findall(a)
        # tag =[w.lower() for w in word_tokenize(a)]
        tag = [w.lower() for w in tag_list]
        print(tag)
        word_list = [pt_stemmer.stem(w) for w in tag]
        print(word_list)
    '''

