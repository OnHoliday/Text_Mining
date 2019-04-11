### TASKS:
# remove sentence # limit! done
# only allow combinations that do not include duplicates. done
# what if topic #1 does not have <270 among quantile? take next best. done (?)
import operator
from collections import namedtuple
import spacy
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;
import gensim 
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import dump, load
import sqlite3
from gensim import matutils
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
import re
from gensim import similarities
import jsonlines
from pprint import pprint
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora
import os
from tempfile import gettempdir
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim import models
import glob
import itertools
from itertools import combinations, product
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity    
import operator
from gensim.models import KeyedVectors
import  logging
from scipy import spatial
from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
from nltk.tokenize import RegexpTokenizer
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import nltk
from gensim.summarization.textcleaner import split_sentences
from gensim.summarization.summarizer import summarize, summarize_corpus
import seaborn as sns
import matplotlib.pyplot as plt


def test():
    print('hi')

class SQL:
    def __init__(self, database):
        self.database = database
        self.conn = sqlite3.connect(database)
    
    def drop_database(self,db_name):
        self.conn.execute('drop database ' + db_name + ' ;')
        
    def create_database(self, db_name):
        self.conn.execute('drop table if exists ' + db_name + ' ;')
        self.conn.execute('CREATE TABLE ' + db_name + ' (content TEXT, title TEXT)')
        
    def import_data(self, db_name,  df):   
        df.to_sql(db_name, self.conn, if_exists='append', index=False)
    
    def get_sample(self, sample_size, table = 'articles'):
        import random
        import pandas as pd
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM " + str(table) +  " LIMIT" + str(sample_size) + ";")
         
        rows = cur.fetchall()
        length = len(rows)
        random_init = random.randint(sample_size, length)
    
        if random_init+sample_size > length:
            sample = rows[random_init-sample_size:random_init]
        else:    
            sample = rows[random_init:random_init+sample_size]
        
        df_articles = pd.DataFrame(sample, columns = ['content', 'title'])
        
        return df_articles
    
    def get_sample_DOMINIKA(self, sample_size, table = 'articles'):
        import random
        import pandas as pd
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM " + str(table) +  " LIMIT" + str(sample_size) + ";")
         
        rows = cur.fetchall()
        
        df_articles = pd.DataFrame(rows, columns = ['content', 'title'])
        
        return df_articles  
    
    def get_all(self, table = 'articles'):
        import pandas as pd
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM " + str(table) + ";")
         
        rows = cur.fetchall()
        df_articles = pd.DataFrame(rows, columns = ['content', 'title'])   
        return df_articles
    
    def len_of_db(self):
        cur = self.conn.cursor()
        cur.execute('SELECT count(*) FROM articles;')
        rows = cur.fetchall()
        return int(rows[0][0])

def split_into_sentences(text):
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
    
def process_raw(text):
    
#    print(text)
    clean = text.lower() 
    clean = re.sub(r"[-+]?[.]?[\d]+[\.]?\d*", "", clean)
    clean = re.sub(r"([a-z])\1{2,}",r"\1\1",clean)
    clean = re.sub(r"(=?[A-Za-z]-{1,})|(=?-{1,}[A-Za-z])","",clean)
    clean = re.sub(r"[\_\-]{1,}", "", clean) 
    clean = re.sub(r'@+[A-Za-z0-9\_\-\?\+\.]*', '', clean)
    clean = re.sub(r'http+.[^\s]*', '', clean)
    
    
    tokenizer = RegexpTokenizer(r'\w+')
    clean = tokenizer.tokenize(clean)
    normalized = [i for i in clean if not i in set(stopwords.words("english"))]
    stemmer= PorterStemmer()
    stemmed_list = [stemmer.stem(word) for word in normalized]
    return stemmed_list

def get_similarity(lda, q_vec, corpus):
    index = similarities.MatrixSimilarity(lda[corpus])
    sims = index[q_vec]
    return sims

def get_nmf_topics(model, vectorizer, n_top_words):
        
        #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
        feat_names = vectorizer.get_feature_names()
        num_topics = 3
        word_dict = {};
        for i in range(num_topics):
            
            #for each topic, obtain the largest values, and add the words they map to into the dictionary.
            words_ids = model.components_[i].argsort()[:-n_top_words - 1:-1]
            words = [feat_names[key] for key in words_ids]
            word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
        
        return pd.DataFrame(word_dict);
    
    
def keywithmaxval(d):
        v = list(d.values())
        k = list(d.keys())
    
        return k[v.index(max(v))]
    
def get_best_sents(sims_,quantile):
    best_sents = []
    for topic in sims_:
        topic_sents = []
        for i in range(len(topic)):
            if topic[i][1] > np.quantile(np.matrix(topic)[:,1], quantile):
                topic_sents.append(topic[i])
        best_sents.append(topic_sents)
    return best_sents


def get_best_combo(best_sents,sims_,num_of_char):
    combos = list(itertools.product(*best_sents))
    combo_stats = []
    for i in combos:
        s = sum(x[1] for x in i)
        l = sum(x[2] for x in i)
        combo_stats.append((s, l))

    if not any(i[1] <= num_of_char for i in combo_stats):
        best_sents = best_sents[0:2]  #

        combos = [list(tup) for tup in itertools.product(*best_sents)]
        # combos = list(itertools.product(*best_sents))

    # Only keeping combo sets!
    combos_set = []
    for combo in combos:
        indexes = [i[0] for i in combo]
        if len(set(indexes)) >= len(indexes):
            combos_set.append(combo)

    combo_stats = []
    for i in combos_set:
        s = sum(x[1] for x in i)
        l = sum(x[2] for x in i)
        combo_stats.append((s, l))

    if not any(i[1] <= num_of_char for i in combo_stats):
        best_sents = best_sents[0]
        best_combo = [best_sents[0]]

        # what if in topic #1 there's still no sentence <270 ?
        if not best_combo[0][2] <= num_of_char:
            best_combo = []
            while len(best_combo) < 1:
                for sent in sims_[0]:
                    if sent[2] <= num_of_char:
                        best_combo.append(sent)


    else:
        best_combo = max(filter(lambda a: a[1] <= num_of_char, combo_stats), key=itemgetter(0))

    for i, v in enumerate(combo_stats):
        if v == best_combo:
            best_combo = combos_set[i]
            break

    if not isinstance(best_combo, list):
        best_combo = sorted(best_combo, key=lambda x: x[0])

    return best_combo

def create_summary(best_combo,sentences):
    # if isinstance(best_combo,list):
    summary = [sentences[best_combo[i][0]] for i in range(len(best_combo))]
    summar = ''
    for i in summary:
        summar += ' '+i

    return summar
    
def get_max_sim_sent(qvec, sent_corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity    
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sent_corpus)
    
    sents =[]
    for i in range(len(qvec)):
        sim_Dic = {}
        a = vectorizer.transform([qvec[i]])
        for j in range(X.shape[0]):
            b = X[j]
            sim_Dic[j] = cosine_similarity(a,b)[0][0]
        sents.append(sent_corpus[keywithmaxval(sim_Dic)])
    return sents


def most_similar(similarity_list, number_of_top):
        top = []
        nr = 0
        for doc in similarity_list:
            nr+=1
            if nr <= number_of_top:
                top.append(doc)
            else: return top
        return top
    
def get_article(url):

    import requests
    import urllib.request
    import time
    from bs4 import BeautifulSoup
    import re
    
    #url = 'https://www.forbes.com/sites/simonmoore/2019/03/23/the-yield-curve-just-inverted-putting-the-chance-of-a-recession-at-30/#6c6b006813ab'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    inside = []
    for i in range(len(soup.findAll('p'))):
        paragraph = str(soup.findAll('p')[i])
        paragraph_clean = re.sub(r"\<(.*?)\>", "", paragraph)
        inside.append(paragraph_clean)
    title = inside[0]
    content = inside[1:]
    content_str = ' '.join(content)
    #return title, content_str
    return  title, content_str

    

def make_tweet(summary,link):
    import requests
    from bs4 import BeautifulSoup
    summary = summary+' '+link
    
    cookies = {
        '_ga': 'GA1.2.1119030962.1542906682',
        'eu_cn': '1',
        'tfw_exp': '0',
        'syndication_guest_id': '1/E0gIAAEAABvqCgACAAABaW7NEq4A/IHzn-aXiEMTggRZ22aRq5fmEdZZUKKIJN2motF31gdVMmQ',
        '_gid': 'GA1.2.1506609955.1553450350',
        'dnt': '1',
        'kdt': 'rTZw9T9vLZl8x1aqRvN2ECZdLWO1lIuWdPgUw9aC',
        'remember_checked_on': '0',
        'csrf_same_site_set': '1',
        'csrf_same_site': '1',
        'mbox': 'PC#ce834bf757b74b7c84f2e970e4c8fc16.26_17#1554761431|session#8bba950464d04c359f296cf6d44ef304#1553553691|check#true#1553551891',
        '__utma': '43838368.1119030962.1542906682.1553478399.1553520034.2',
        '__utmz': '43838368.1553478399.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none)',
        'lang': 'en',
        'ct0': '36773414da7dce7693d2b2729e1a3703',
        '_twitter_sess': 'BAh7CiIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCMAR37ZpAToMY3NyZl9p%250AZCIlMWI4NDgwYWRkOTQzZjQ3OTEzN2ZkNDgxMGE4ZjhiZTA6B2lkIiUyMDg0%250ANzUyOThmY2MyN2NjNGVlZjZkMDdkNzIwYTUyMToJdXNlcmwrCQFw17oG%252FmYP--55266aeb323113b7a7dff34400b6188448ae6815',
        'personalization_id': 'v1_Glw8cQZCkDTtz7LxHDk6RQ==',
        'guest_id': 'v1%3A155355126623799447',
        'ads_prefs': 'HBISAAA=',
        'twid': 'u=1109853663051345921',
        'auth_token': '9370b813a52ae1eabff7088a438f3417074eb524',
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'pl,en-US;q=0.7,en;q=0.3',
        'Referer': 'https://twitter.com/?lang=en',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'X-Twitter-Active-User': 'yes',
        'X-Requested-With': 'XMLHttpRequest',
        'Connection': 'keep-alive',
        'TE': 'Trailers',
    }
    
    data = {
      'authenticity_token': 'ad9941f0ea9bc24156a149a220e82fb21ee76860',
      'batch_mode': 'off',
      'is_permalink_page': 'false',
      'lang': 'en',
      'place_id': '',
      'status': 'finally it is working!!!',
      'tagged_users': ''
    }
    
    username = "Group12NOVA1"
    password = "Group124eva"
    # login url
    post = "https://twitter.com/sessions"
    url = 'https://twitter.com'
    
    data = {"session[username_or_email]": username,
            "session[password]": password,
            "scribe_log": "",
            "redirect_after_login": "/",
            "remember_me": "1"}
    
    r = requests.get(url)
        # get auth token
    soup = BeautifulSoup(r.content, "lxml")
    AUTH_TOKEN = soup.select_one("input[name=authenticity_token]")["value"]
    
    
    data["authenticity_token"] = AUTH_TOKEN
#    tweet = 'now i can do whatever i want , i am the king of the world!!!+#king+@whoever'
    data["status"] = summary
    
    
    ### make tweet!
    response = requests.post('https://twitter.com/i/tweet/create', headers=headers, cookies=cookies, data=data)



# =============================================================================
# def ElmoEmbedding(x):
#     elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
#     max_len = 50
#     batch_size = 32
# 
#     with open('tags.pkl', 'rb') as f:
#         tags = pickle.load(f)
#     n_tags=len(tags)
#     return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x,    tf.string)),"sequence_len": tf.constant(batch_size*[max_len])
#                      },
#                       signature="tokens",
#                       as_dict=True)["elmo"]
# 
# =============================================================================
def twitterize(text):
#TODO decide on which entities to put which hashtag    https://medium.com/explore-artificial-intelligence/introduction-to-named-entity-recognition-eda8c97c2db1
    ''' takes a string and recognises its entities. on some entities a hashtag or @ is placed before 
    the word to make the text ready for a human like tweet'''
    
    #using the spacy library
    try:
        nlp.toggleUI()
    except NameError:
        nlp = spacy.load("en_core_web_sm")# Load English tokenizer, tagger, parser, NER and word vectors
    doc = nlp(text) # performing the NER
    
   
    x = [[X, X.ent_iob_, X.ent_type_] for X in doc][::-1] # get a list of lits with the word iob and entity type for every token in the sentence in reverse order
    
    #joining Entities  PERSON ORG NORP if they consits of multiple words
    for i, lst in enumerate(x): 
        
        if lst[2] in list([str('PERSON'), str('ORG'), str('NORP'), str('FAC'), str('GPE'), str('LOC'), str('PRODUCT'), str('EVENT'), str('WORK_OF_ART'), str('LAW'), str('LANGUAGE')]):
          
            if lst[1] == 'I':
                x[i+1][0] =  str(x[i+1][0]) + str(x[i][0])
                x[i]='double'
    x=[a for a in x if a != 'double'][::-1]
                
    
    
    for i in range(len(x)):
        if x[i][2] in list([str('NORP'), str('FAC'), str('GPE'), str('LOC'), str('PRODUCT'), str('EVENT'), str('WORK_OF_ART'), str('LAW'), str('LANGUAGE')]):
    
            word = x[i][0]
     
            word = '#'+str(word) # putting a string infront of the word
            
            x[i][0] = word
            
        
        if x[i][2] in list([str('PERSON'), str('ORG')]):
            word = x[i][0]
     
            word = '@'+str(word) # putting a string infront of the word
            
            x[i][0] = word
    y =[]
    
    for i in x:
        y.append(str(i[0]))
    y = " ".join(y) # joining all the words
    y = re.sub(r'\s*(?=[:,; .])',"",y)
    y = re.sub(r'\s*(?=[’\'][a-z]{1,2})',"",y)
    y = re.sub(r's*(?=n[\'’]t)',"",y)
    y = re.sub(r'(?<=\()\s*(?=[A-Za-z])',"",y)
    y = re.sub(r'(?<=[A-Za-z])\s*(?=\))',"",y)

    return y


#def twitterize_ELMO(text):
#    
#    
#    ''' takes a string and recognizes its entitites. on some entites a hastag or @ is placed before 
#    the word to make the text ready for a human like tweet'''
#    max_len = 50
#    batch_size = 32
#
#    with open('tags.pkl', 'rb') as f:
#        tags = pickle.load(f)
#    n_tags=len(tags)
#    input_text = Input(shape=(max_len,), dtype=tf.string)
#    embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
#    x = Bidirectional(LSTM(units=512, return_sequences=True,
#                       recurrent_dropout=0.2, dropout=0.2))(embedding)
#    x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
#                           recurrent_dropout=0.2, dropout=0.2))(x)
#    x = add([x, x_rnn])  # residual connection to the first biLSTM
#    out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)
#    model_2 = Model(input_text, out)
#    model_2.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#    model_2.load_weights("my_model_weights.h5")
#    
#    
#    
#    tokens = np.array(nltk.word_tokenize(text))
#
#    new_seq=[]
#    for i in range(max_len):
#        try:
#            new_seq.append(tokens[i])
#        except:
#            new_seq.append("PADword")
#    sentence =new_seq
#    
#    sentences = []
#    for i in range(batch_size):
#        sentences.append(sentence)
#        
#    
#    p = model_2.predict(np.array(sentences))[0]
#    p = np.argmax(p, axis=-1)
#    
#    pred = [[word,  tags[ent_type]] for word,  ent_type in zip(sentences[0], p)][::-1] # get a list of li        
#
#
#    matches_I = [x for x in tags if re.match(r'I',x)]
#    matches_R = [x for x in tags if not re.match(r'[IO]',x)]
#    
#    
#    #joining Entities  PERSON ORG NORP if they consits of multiple words
#    for i, item in enumerate(pred): 
#        
#        if item[1] in matches_I and (pred[i+1][1] in matches_R or pred[i+1][1] in matches_I) :
##            print(lst[1])
##            print(lst[1])
#            pred[i+1][0] =  str(pred[i+1][0]) + str(pred[i][0])
#            pred[i]='double'
#    pred=[a for a in pred if a != 'double'][::-1]
#    pred=[a for a in pred if a[0] != "PADword"]
#                
#    
#    
#    for i in range(len(pred)):
#        if not pred[i][1] == "O":
#    
#            word = pred[i][0]
#     
#            word = '#'+str(word) # putting a string infront of the word
#            
#            pred[i][0] = word
#    tweet =[]
#    
#    for i in pred:
#        tweet.append(str(i[0]))
#    tweet = " ".join(tweet) # joining all the words
#    tweet = re.sub(r'\s*(?=[:,; .])',"",tweet)
#    tweet = re.sub(r'\s*(?=[’\'][a-z]{1,2})',"",tweet)
#    tweet = re.sub(r's*(?=n[\'’]t)',"",tweet)
#    tweet = re.sub(r'(?<=\()\s*(?=[A-Za-z])',"",tweet)
#    tweet = re.sub(r'(?<=[A-Za-z])\s*(?=\))',"",tweet)
#    return tweet


def plot_with_labels(low_dim_embs, low_dim_embs_words, labels,df_sum, filename):
    # create dot for title and summary
    plt.figure(figsize=(8, 8))  # in inches
    for i, label in enumerate(labels[0]):
        print(label)
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')
    # connecting lines
    plt.plot([low_dim_embs[0][0], low_dim_embs[0][0]], [low_dim_embs[0][1], low_dim_embs[0][1]])
    plt.plot([low_dim_embs[0][0], low_dim_embs[1][0]], [low_dim_embs[0][1], low_dim_embs[1][1]],
             linewidth=int(df_sum['sim'][3] * 10))
    plt.plot([low_dim_embs[0][0], low_dim_embs[2][0]], [low_dim_embs[0][1], low_dim_embs[2][1]],
             linewidth=int(df_sum['sim'][4] * 10))
    plt.plot([low_dim_embs[0][0], low_dim_embs[3][0]], [low_dim_embs[0][1], low_dim_embs[3][1]],
             linewidth=int(df_sum['sim'][5] * 10))
    plt.plot([low_dim_embs[0][0], low_dim_embs[4][0]], [low_dim_embs[0][1], low_dim_embs[4][1]],
             linewidth=int(df_sum['sim'][6] * 10))
    
    for i, label in enumerate(labels[1]):
        x, y = low_dim_embs_words[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')
