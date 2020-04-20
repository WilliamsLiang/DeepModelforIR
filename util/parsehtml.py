import re
import nltk.stem.porter as pt

from bs4 import BeautifulSoup
from util.basefunction import filter_tags

pt_stemmer = pt.PorterStemmer()
pat_letter = re.compile(r'[a-zA-Z\-\']+')


class HtmlParseBybs:
    def __init__(self):
        self.soupstring=None
        pass

    def parse(self,html):
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
        self.tagre=re.compile(r"<"+tag+".*?</"+tag+">")
        tag_list=[filter_tags(_) for _ in self.tagre.findall(self.htmlstring)]
        return tag_list

    def parse_get(self,html,tag):
        """
        :param html:带解析的HTML 字符串
        :param tag: 获取的标签
        :return:  解析标签之后的文本内容
        """
        self.htmlstring=self.scriptre.sub("",html)
        self.tagre = re.compile(r"<" + tag + ".*?</" + tag + ">")
        tag_list = [filter_tags(_) for _ in self.tagre.findall(self.htmlstring)]
        return tag_list


if __name__=="__main__":
    f=open("E:/CLUBWEB_experiment/ntcir_www2/2.txt","rt",encoding="utf-8")
    line=f.readline()
    pt_stemmer = pt.PorterStemmer()
    datas=line.split("\t")
    html_code=datas[-1]
    f.close()
    a_list=HtmlParseBybs().parse_get(html_code,"body")
    for a in a_list:
        print(a)
        tag_list=pat_letter.findall(a)
        #tag =[w.lower() for w in word_tokenize(a)]
        tag = [w.lower() for w in tag_list]
        print(tag)
        word_list=[pt_stemmer.stem(w) for w in tag]
        print(word_list)
    a_list = HtmlParseByre().parse_get(html_code, "body")
    for a in a_list:
        print(a)
        tag_list = pat_letter.findall(a)
        # tag =[w.lower() for w in word_tokenize(a)]
        tag = [w.lower() for w in tag_list]
        print(tag)
        word_list = [pt_stemmer.stem(w) for w in tag]
        print(word_list)