# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:19:26 2019

@author: Konrad
"""



from combined_utils_4 import *

# load the google word2vec model
model = KeyedVectors.load_word2vec_format(r'D:\word2vec_model\GoogleNews-vectors-negative300.bin', binary=True)
model = gensim.models.Word2Vec.load('mymodel')

# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['man', 'boy'], negative=['woman'], topn=1)

print(result)


from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath(r'C:\Users\Konrad\Desktop\NOVA IMS\text mining\Project\glove.6B.300d.txt')
tmp_file = get_tmpfile("test_word2vec.txt")
_ = glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)

result = model.most_similar(positive=['paris', 'italy'], negative=['france'], topn=1)[0][0]
print('\nResult of the query is: %s' % result[0][0].capitalize())

result = model.wv.most_similar(['red'], topn=8)


result = model.wv.most_similar(positive=['one', 'two'], topn=1)

model.most_similar(negative=['ten', 'five'], topn=1)

model.wv.accuracy(r'C:\Users\Konrad\Desktop\NOVA IMS\text mining\evaluete_accuracy_dataset.txt')


model = gensim.models.Word2Vec.load('mymodel')
acc = model.accuracy(r'C:\Users\Konrad\Desktop\NOVA IMS\text mining\evaluete_accuracy_dataset.txt')

