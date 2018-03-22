# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 21:32:16 2018

@author: Deep
"""
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import collections
import numpy as np
from keras.preprocessing.sequence import pad_sequences

english_stopwords = stopwords.words('english')

# 缩写字符的替换和无意义字符的去除
def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    return string.strip().lower()


# 读取数据，得到[story, question, answer, correct]四元组的分词形式
def get_sqac_pairs_tokenizes(qa_dir, ins_dir):  
    
    # 得到[story_idx, question, answer, correct]的四元组
    data_qa = list(open(qa_dir,'r'))

    qac_pairs = []
    qac_list = []
    
    for line in data_qa:
        line = line.strip()
        line = line.split("\t")
        del line[1]
        qac_list.append(line)
      
    for item in qac_list:
        story_idx = item[0]
        question = item[1]
        answer = []
        answer1 = item[2]
        answer2 = item[3]
        correct = int(item[4])
        answer.append(answer1)
        answer.append(answer2)
        for a in answer:
            torf = True
            if ((correct == 0) and (a == answer[0])):
                torf = True
            elif ((correct == 0) and (a == answer[1])):
                torf = False
            elif (a == answer[0]):
                torf = False
            else:
                torf = True
            qac_pairs.append([story_idx, question, a, torf])


    # 得到[story, question, answer, correct]的四元组           
    data_instance = list(open(ins_dir,'r'))
    story = []
    sqac_pairs = []
    for line in data_instance:
        line = line.strip()
        line = line.split("\t")
        story.append(line)
    for item in story:
        for item1 in qac_pairs:
            if item1[0] == item[0]:
                sqac_pairs.append([item[1], item1[1], item1[2], item1[3]])
                      
    # 得到四元组的分词形式
    sqac_pairs_tokenizes = []
    for item in sqac_pairs:
        temp = clean_str(item[0])
        item[0] = temp
        story = item[0]
        temp = clean_str(item[1])
        item[1] = temp
        question = item[1]
        temp = clean_str(item[2])
        item[2] = temp
        answer = item[2]
        
        # 分词
        tt = TweetTokenizer()
        swords = tt.tokenize(story)
        qwords = tt.tokenize(question)
        awords = tt.tokenize(answer)
        
        # 去停用词，只对story去停用词
        swords = [sw for sw in swords if not sw in english_stopwords]
#        qwords = [qw for qw in qwords if not qw in english_stopwords]
#        awords = [aw for aw in awords if not aw in english_stopwords]
        
        # 去标点
#        swords = [sw for sw in swords if not sw in english_punctuations]
#        qwords = [qw for qw in qwords if not qw in english_punctuations]
#        awords = [aw for aw in awords if not aw in english_punctuations]
                                
        # 词干化
#        swords = [lemmatizer.lemmatize(sw) for sw in swords]
#        swords = [lemmatizer.lemmatize(sw, pos='v') for sw in swords]
#        
#        qwords = [lemmatizer.lemmatize(qw) for qw in qwords]
#        qwords = [lemmatizer.lemmatize(qw, pos='v') for qw in qwords]
#        
#        awords = [lemmatizer.lemmatize(aw) for aw in awords]
#        awords = [lemmatizer.lemmatize(aw, pos='v') for aw in awords]
        
        is_correct = item[3]
        
        sqac_pairs_tokenizes.append([swords, qwords, awords, is_correct])
        
    return sqac_pairs_tokenizes 

"""
测试代码，返回的是[[story的分词形式], [question的分词形式], [answer的分词形式], [True/False]]的四元组形式
qa_dir = "C:/Users/Deep/Desktop/SemEval t11/data/train_qa.txt"
ins_dir = "C:/Users/Deep/Desktop/SemEval t11/data/train_ins.txt"
sqac_pairs_tokenizes = get_sqac_pairs_tokenizes(qa_dir, ins_dir)
"""

# 建立sqa 3元组的词表, 索引从1开始，0作为mask
def build_vocab_from_sqac_pairs_tokenizes(sqac_pairs_tokenizes):
    wordcounts = collections.Counter()
    for sqatriple in sqac_pairs_tokenizes:
        for sword in sqatriple[0]:
            wordcounts[sword] += 1
        for qword in sqatriple[1]:
            wordcounts[qword] += 1
        for aword in sqatriple[2]:
            wordcounts[aword] += 1
    words = [wordcount[0] for wordcount in wordcounts.most_common()]
    word2idx = {w: i+1 for i, w in enumerate(words)}  # 0 = mask
    return  word2idx

"""
测试代码，返回{word:idx}的字典，idx越小代表该词出现的频率越高，索引从1开始，0作为mask
sqac_pairs_tokenizes = get_sqac_pairs_tokenizes(qa_dir, ins_dir)
word2idx = build_vocab_from_sqac_pairs_tokenizes(sqac_pairs_tokenizes)
"""

# 得到story, question, answer的向量化表示(以词表中的value值作为一个word的数字表示，值越小代表出现的频率越高)
# 并以各自的最大长度作为统一长度，不够的在句子后面补0   
# 返回Xs(story的向量表示), Xq(question的向量表示)，Xa(answer的向量表示), Y(samples*2的矩阵，[1,0]代表True,反之False)
def vectorize_sqac_pairs_tokenizes(sqac_pairs_tokenizes, word2idx, story_maxlen, 
                         question_maxlen, answer_maxlen):
    Xs, Xq, Xa, Y = [], [], [], []
    for sqatriple in sqac_pairs_tokenizes:
        Xs.append([word2idx[sword] for sword in sqatriple[0]])
        Xq.append([word2idx[qword] for qword in sqatriple[1]])
        Xa.append([word2idx[aword] for aword in sqatriple[2]])
        Y.append(np.array([1, 0]) if sqatriple[3] else np.array([0, 1]))
        
    return (pad_sequences(Xs, maxlen=story_maxlen, padding="post"),
            pad_sequences(Xq, maxlen=question_maxlen, padding="post"),
            pad_sequences(Xa, maxlen=answer_maxlen, padding="post"),
            np.array(Y))
"""
测试代码
sqac_pairs_tokenizes = get_sqac_pairs_tokenizes(qa_dir, ins_dir)
word2idx = build_vocab_from_sqac_pairs_tokenizes(sqac_pairs_tokenizes)
story_maxlen = max([len(sqatriple[0]) for sqatriple in sqac_pairs_tokenizes])
question_maxlen = max([len(sqatriple[1]) for sqatriple in sqac_pairs_tokenizes])
answer_maxlen = max([len(sqatriple[2]) for sqatriple in sqac_pairs_tokenizes])

Xs, Xq, Xa, Y = vectorize_sqac_pairs_tokenizes(sqac_pairs_tokenizes, word2idx, story_maxlen, 
                                               question_maxlen, answer_maxlen)
"""


























