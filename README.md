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

    ltrnorm.py 计算查询式和文档之间的相似度

### util
    basefunction.py 存放基础函数

    parsehtml.py
    解析函数依赖库BeautigulSoup4 nltk 
    HtmlParseBybs 使用BS4解析HTML代码，提取主要内容。（推荐）
    HtmlParseByre 使用正则表达式，提取主要内容（解决BS4解析错误的情况）
    HtmlInfo 用HtmlParseBybs解析代码，并提取token（词形还原后）
    
    bigcount.py
    计算全局文档的tfidf 和 P(w|c) 并用于BigBM25和BigLMIR

### main
    主函数模块，不用于封装功能
    countfidf.py 计算特征
    ltr_main.py ltr 训练和预测代码。
    

### BASE_EVALTOOL(拓展工具)
    官方提供的评价工具,C语言编写。
    

## 2.数据资源
    数据资源分为中文子任务和英文子任务
    中文数据地址文件夹：/home/user/ntcir_match/zh_task
    英文数据地址文件夹：/home/user/ntcir_match/en_task
    
    模型特征说明文档：LambdaMart模型特征集.xlsx

## 3.相关文献
    1.LMIR:A Study of Smoothing Methods for Language Models Applied to Information Retrieval[J].
    2.DSSM:Learning deep structured semantic models for web search using clickthrough data[C].
    3.CDDM:Learning Semantic Representations Using Convolutional Neural Networks for Web Search[C].
    
## 版本控制
##### 控制说明 版本号：时间_更新者缩写_版次 例如（20200420_lz_01）
### 20200420_lz_01
    更新基础函数、评价工具以及基本的信息检索系统框架

### 202000509_lz_01
    1.修正了HtmlParseBybs,可以提取网址代码。增加HtmlInfo类，用以处理HTML代码，并返回token
    2.新增ltrnorm.py,主函数countidf.py、ltr_main.py,util模块 bigcount.py
    (ps. learning to rank 进行无监督训练时 标签定位1，0 评价函数为map)
    3.修正 trecEval 新增方法 get_NDCGfilter评价相关性文档在全局的排序水平（NTCIR使用的get_avgNDCG）
    
    ltr模型结果： NDCG@10 0.28599 
    参数：评价函数MAP 正反样本比1：3 
    n_estimators=5000 learning_rate=0.02 max_features=0.5 query_subsample=0.5 max_leaf_nodes=10 min_samples_leaf=64 verbose=1 
    （ltrnorm 输出的样本不是按qid group 后续利用Linux 命令排序 sort -n -k 2.5,2.8 filename）

### 20210823_lz_01
    增加了BERTforIR、DSSM、DRMM模型
