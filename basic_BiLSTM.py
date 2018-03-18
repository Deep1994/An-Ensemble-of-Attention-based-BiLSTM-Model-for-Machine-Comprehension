# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:10:48 2018

@author: Deep

"""
import dataProcess
import testReader
from gensim.models import KeyedVectors
import os
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, Merge, Dropout, Flatten
from keras.layers.recurrent import LSTM

TRAIN_QA_DIR = "C:/Users/Deep/Desktop/SemEval t11/data/train_qa.txt" 
TRAIN_INS_DIR = "C:/Users/Deep/Desktop/SemEval t11/data/train_ins.txt"
DEV_QA_DIR = "C:/Users/Deep/Desktop/SemEval t11/data/dev_qa.txt"
DEV_INS_DIR = "C:/Users/Deep/Desktop/SemEval t11/data/dev_ins.txt"
TEST_QA_DIR = "C:/Users/Deep/Desktop/gold_data/test_data_qa.txt"
TEST_INS_DIR = "C:/Users/Deep/Desktop/gold_data/test_data_ins.txt"
WORDVEC_DIR = "C:/Users/Deep/Desktop/SemEval t11/w2v"
#WORD2VEC_BIN = "GoogleNews-vectors-negative300.bin"
WORD2VEC_BIN = "glove.840B.300d.txt"

WORD2VEC_EMBED_SIZE = 300
QA_EMBED_SIZE = 64
BATCH_SIZE = 512
EPOCHS = 6
seed = 7
np.random.seed(seed)

# 加载数据
print("Loading and formatting data...")
sqac_pairs_tokenizes = dataProcess.get_sqac_pairs_tokenizes(TRAIN_QA_DIR, TRAIN_INS_DIR)
dev_sqac_pairs_tokenizes = dataProcess.get_sqac_pairs_tokenizes(DEV_QA_DIR, DEV_INS_DIR)
test_sqac_pairs_tokenizes = testReader.get_sqac_pairs_tokenizes(TEST_QA_DIR, TEST_INS_DIR)
all_pairs_tokenizes = sqac_pairs_tokenizes + dev_sqac_pairs_tokenizes + test_sqac_pairs_tokenizes
story_maxlen = max([len(sqatriple[0]) for sqatriple in all_pairs_tokenizes])
question_maxlen = max([len(sqatriple[1]) for sqatriple in all_pairs_tokenizes])
answer_maxlen = max([len(sqatriple[2]) for sqatriple in all_pairs_tokenizes])

word2idx = dataProcess.build_vocab_from_sqac_pairs_tokenizes(all_pairs_tokenizes)
vocab_size = len(word2idx) + 1 # include mask character 0

# 训练集
Xs, Xq, Xa, Y = dataProcess.vectorize_sqac_pairs_tokenizes(sqac_pairs_tokenizes, word2idx, story_maxlen, question_maxlen, answer_maxlen)

# 开发集
Xs_dev, Xq_dev, Xa_dev, Y_dev = dataProcess.vectorize_sqac_pairs_tokenizes(dev_sqac_pairs_tokenizes, word2idx, story_maxlen, question_maxlen, answer_maxlen)

# 测试集
Xs_test, Xq_test, Xa_test, Y_test = testReader.vectorize_sqac_pairs_tokenizes(test_sqac_pairs_tokenizes, word2idx, story_maxlen, question_maxlen, answer_maxlen)

# 将训练集和开发集合并成为新的训练集，19674 + 2834 = 22508
#Xs_all = np.vstack((Xs, Xs_dev))
#Xq_all = np.vstack((Xq, Xq_dev))
#Xa_all = np.vstack((Xa, Xa_dev))
#Y_all = np.vstack((Y, Y_dev))


# get embeddings from word2vec
print("Loading Word2Vec model and generating embedding matrix...")
#word2vec = KeyedVectors.load_word2vec_format(os.path.join(WORDVEC_DIR, WORD2VEC_BIN), binary=True)
word2vec = KeyedVectors.load_word2vec_format(os.path.join(WORDVEC_DIR, WORD2VEC_BIN), binary=False, unicode_errors='ignore')

embedding_weights = np.zeros((vocab_size, WORD2VEC_EMBED_SIZE))
num_unknownwords = 0
unknow_words = []
for word, index in word2idx.items():
    try:
        embedding_weights[index, :] = word2vec[word]
    except KeyError:
        num_unknownwords += 1
        unknow_words.append(word)
        embedding_weights[index, :] = np.random.uniform(-0.25, 0.25, WORD2VEC_EMBED_SIZE) 

# print(word2vec.wv.vocab.__len__())

##########################################################################################################
        
print("Building model...")

# story encoder.
# output shape: (None, story_maxlen, QA_EMBED_SIZE)
senc = Sequential()
senc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=story_maxlen,
                   weights=[embedding_weights], mask_zero=True))
#senc.add(LSTM(QA_EMBED_SIZE, return_sequences=True))
senc.add(Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=True), 
                       merge_mode="sum"))
senc.add(Dropout(0.3))

# question encoder
# output shape: (None, question_maxlen, QA_EMBED_SIZE)
qenc = Sequential()
qenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=question_maxlen,
                   weights=[embedding_weights], mask_zero=True))
#qenc.add(LSTM(QA_EMBED_SIZE, return_sequences=True)) 
qenc.add(Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=True), 
                       merge_mode="sum"))
qenc.add(Dropout(0.3))

# answer encoder
# output shape: (None, answer_maxlen, QA_EMBED_SIZE)
aenc = Sequential()
aenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=answer_maxlen,
                   weights=[embedding_weights], mask_zero=True))
#aenc.add(LSTM(QA_EMBED_SIZE, return_sequences=True)) 
aenc.add(Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=True), 
                       merge_mode="sum"))                  
aenc.add(Dropout(0.3))

# merge story and question => facts
# output shape: (None, story_maxlen, question_maxlen)
facts = Sequential()
facts.add(Merge([senc, qenc], mode="dot", dot_axes=[2, 2]))

# merge question and answer => attention
# output shape: (None, answer_maxlen, question_maxlen)
attn = Sequential()
attn.add(Merge([aenc, qenc], mode="dot", dot_axes=[2, 2]))

# merge facts and attention => model
# output shape: (None, story+answer_maxlen, question_maxlen)
model = Sequential()
model.add(Merge([facts, attn], mode="concat", concat_axis=1))
model.add(Flatten())
model.add(Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])


print("Training...")
model.fit([Xs, Xq, Xa], Y, batch_size=BATCH_SIZE, epochs=EPOCHS)

predict_Y = model.predict([Xs_dev, Xq_dev, Xa_dev]) 
# 第0个答案偏向正确的概率
a_0 = (predict_Y[::2])[:,0]
# 第1个答案偏向正确的概率
a_1 = (predict_Y[1::2])[:,0]

# 自己动手计算准确率   
# asw是提交文件的第三列
asw = []
for a in zip(a_0, a_1):
    #print (i)
    if a[0] > a[1]:
        asw.append(0)
    else:
        asw.append(1)

aidx = []
dev_qa = list(open(DEV_QA_DIR, 'r'))
for line in dev_qa:
     line = line.strip()
     line = line.split("\t")
     del line[0:5]
     aidx.append(line)
 
selfEva = []    
for t in zip(aidx, asw):
    #print (t)
    if int(t[0][0]) == t[1]:
        selfEva.append(1)
    else:
        selfEva.append(0)
        
num_correct = 0       
for i in selfEva:
    if i == 1:
        num_correct += 1
self_accuracy = num_correct/len(selfEva)
print (self_accuracy)  

# 在测试集上的预测
predict_test_1 = model.predict([Xs_test, Xq_test, Xa_test]) 

# 第0个答案偏向正确的概率
test_a_0_1 = (predict_test_1[::2])[:,0]
# 第1个答案偏向正确的概率
test_a_1_1 = (predict_test_1[1::2])[:,0]

# 自己动手计算准确率   
asw = []
for a in zip(test_a_0_1, test_a_1_1):
    #print (i)
    if a[0] > a[1]:
        asw.append(0)
    else:
        asw.append(1)

aidx = []
test_qa = list(open(TEST_QA_DIR, 'r'))
for line in test_qa:
     line = line.strip()
     line = line.split("\t")
     del line[0:6]
     aidx.append(line)
 
selfEva = []    
for t in zip(aidx, asw):
    #print (t)
    if int(t[0][0]) == t[1]:
        selfEva.append(1)
    else:
        selfEva.append(0)
        
num_correct = 0       
for i in selfEva:
    if i == 1:
        num_correct += 1
self_accuracy = num_correct/len(selfEva)
print (self_accuracy)