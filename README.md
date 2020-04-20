# NTCIR www project
#### 本次项目主要针对NTCIR 的子任务WWW所设置
NTCIR WWW的比赛网址：[http://sakailab.com/www3/](http://sakailab.com/www3/)

## 1.模块介绍
### evalfunction
    retrievalEval.py NDCG评价指标
    traditional.py 查全查准率
   
### simmodel
    languageModel.py LMIR模型
    statisticModel.py BM25模型
    
### trecnorm
    normoutput.py 
    normTrec 为TREC标准输出格式
    trecEval 为TREC文件评价类，自动加载相似度文件和结果文件

### util
    basefunction.py 存放基础函数
    parsehtml.py
    解析函数依赖库BeautigulSoup4 nltk 
    HtmlParseBybs 使用BS4解析HTML代码，提取主要内容。（推荐）
    HtmlParseByre 使用正则表达式，提取主要内容（解决BS4解析错误的情况）
    
### BASE_EVALTOOL(拓展工具)
    官方提供的评价工具,C语言编写。
    

## 2.数据资源
    数据资源分为中文子任务和英文子任务
    服务器地址：192.168.116.123
    中文数据地址文件夹：/home/user/ntcir_match/zh_task
    英文数据地址文件夹：/home/user/ntcir_match/en_task
    
    模型特征说明文档：LambdaMart模型特征集.xlsx
    其他资料见群文件：群号码：NRCIR WWW交流群 805627385

## 3.相关文献
    1.LMIR:A Study of Smoothing Methods for Language Models Applied to Information Retrieval[J].
    2.DSSM:Learning deep structured semantic models for web search using clickthrough data[C].
    3.CDDM:Learning Semantic Representations Using Convolutional Neural Networks for Web Search[C].
    
## 版本控制
##### 控制说明 版本号：时间_更新者缩写_版次 例如（20200420_lz_01）
### 20200420_lz_01
    更新基础函数、评价工具以及基本的信息检索系统框架