# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:20:43 2019

@author: Konrad
"""
from combined_utils_4 import *
import gensim 
import pandas as pd
from gensim.models import KeyedVectors
#model = KeyedVectors.load_word2vec_format(r'D:\word2vec_model\GoogleNews-vectors-negative300.bin', binary=True)
model = gensim.models.Word2Vec.load('mymodel')


import  logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


from scipy import spatial

def summarizor(url_app):
    
    url = url_app
#    url = 'https://www.forbes.com/sites/gordonkelly/2019/04/04/apple-iphone-fold-flip-clamshell-iphone-xs-max-xr-upgrade/#211a3c5d4625'
    x = pd.DataFrame(columns=['title', 'content'])
    x.loc[0,['title', 'content']] = get_article(url)

    dictionary = corpora.Dictionary()
    
    # Remove multiple whitespaces
#    for index, row in x.iterrows():
#        row['content']=' '.join(row['content'].split())
    
    # Create corpus dictionary
    dictionary = corpora.Dictionary()
    
    # Define number of top sentences to pick for each topic
    num_of_sents = 3
    num_of_char = 257
    # Define quantile of similarity score the sentences need to fulfill
    quantile= 0.85
    
    # run functions for each article in the corpus
    for index, row in x.iterrows():
        print("Summarizing article ",index +1, "out of", x.shape[0])
        article = row['content']
        sentences = split_into_sentences(row['content'])
        article_clean = [process_raw(sentence) for sentence in sentences]
    
        x.loc[index,'content_clean'] = row['content']
        x.loc[index,'content_clean'] = article_clean
        # creating bag of words for gensim library
        dictionary.add_documents(article_clean)
            
        # create corpus in desired format  
        corpus = [dictionary.doc2bow(sentence) for sentence in article_clean]
        
        # create instance of a class to perform tfidf
        tfidf = models.TfidfModel(corpus)
        
        # perform tfidf
        corpus_tfidf = tfidf[corpus]
        
        # print sample of corpus after performing tfidf
        count = 0
    
        # perform LSI############################################       
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3)
        corpus_lsi = lsi[corpus_tfidf]
        
        V = matutils.corpus2dense(lsi[corpus_tfidf], len(lsi.projection.s)).T / lsi.projection.s
        
        b = np.array(V) # V is a matrix from SVD
        b_list = [list(b[:, i - 1]) for i in range(b.shape[1])]
    
        sims_ = []
        lengths = []
        for i in b_list:
            tuples = [(i.index(value), abs(value), len(sentences[i.index(value)])) for value in i]
            ## LSI = higher abs value = higher topic relation??
            ordered = sorted(tuples, key=lambda x: x[1], reverse=True)
            sims_.append(ordered)
    
        best_sents = get_best_sents(sims_, quantile)
    
        best_combo = get_best_combo(best_sents,sims_,num_of_char)
    
        summar = create_summary(best_combo,sentences)
        
        x.loc[index,'LSI summary'] = row['content']
        x.loc[index,'LSI summary'] = summar
        
        # perform LDA############################################  
        corpus = [dictionary.doc2bow(sentence) for sentence in article_clean]
        NUM_OF_TOPICS = 3
        ldamodel = models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_OF_TOPICS)
        topics = ldamodel.show_topics(formatted=False)
    # =============================================================================
        coherences = {}
        for n, topic in topics:
            topic = [word for word, _ in topic]
            cm = CoherenceModel(topics=[topic], corpus=corpus, coherence='u_mass', dictionary=dictionary)
            coherences[n] = abs(cm.get_coherence())
    
        coh_sorted = sorted(coherences.items(), key=operator.itemgetter(1))
        coh_indexes=[i[0] for i in coh_sorted]
        ordered_topics= [topics[i] for i in coh_indexes]
        ordered_topics = ordered_topics[0:3]
    
        summary = []
        sims_ = []
        for i in range(3):
            topic = ordered_topics[i][1]
            query = ' '.join(word[0] for word in topic)
            bow = dictionary.doc2bow(process_raw(query))
            q_vec = ldamodel[bow]
            sims = get_similarity(ldamodel, q_vec, corpus)
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            lengths = []
            for i in range(len(sims)):
                add_len = sims[i] + (len(sentences[sims[i][0]]),)
                lengths.append(add_len)
            sims_.append(lengths)
    
        best_sents = get_best_sents(sims_, quantile)
    
        best_combo = get_best_combo(best_sents,sims_,num_of_char)
    
        summar = create_summary(best_combo, sentences)
    
        x.loc[index, 'LDA summary'] = row['content']
        x.loc[index, 'LDA summary'] = summar
    
                                                         
        #######3gensim
        vectorizer = CountVectorizer(tokenizer=process_raw, lowercase=True)
        X=vectorizer.fit_transform(sentences)
        df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    
        # Tf-idf matrix
        TfidfVec = TfidfVectorizer(tokenizer=process_raw, lowercase=True)
        matrix = TfidfVec.fit_transform(sentences)
        tfidf_matrix = pd.DataFrame(matrix.toarray(), columns=TfidfVec.get_feature_names())
        
    
        #DOM NO need to normalize, TfidfVec.fit_transform does that already
        xtfidf_norm = normalize(tfidf_matrix, norm='l1', axis=1)
        
        num_topics = 3
        
        #obtain a NMF model.
        modelnmf = NMF(n_components=num_topics, init='nndsvd')
        #fit the model
        modelnmf.fit(xtfidf_norm) 
        
        nmf_topics=get_nmf_topics(modelnmf, vectorizer, 10)
        nmf_topics=nmf_topics.transpose().values.tolist()
        
        coherences = {}
        for i in range(len(nmf_topics)):
            topic = [word for word in nmf_topics[i]]
            cm = CoherenceModel(topics=[topic], corpus=corpus, coherence='u_mass', dictionary=dictionary)
            coherences[i] = abs(cm.get_coherence())
    
        coh_sorted = sorted(coherences.items(), key=operator.itemgetter(1))
        coh_indexes=[i[0] for i in coh_sorted]
        ordered_topics= [nmf_topics[i] for i in coh_indexes]
        ordered_topics = ordered_topics[0:3]
        
        qvec = []
        for topic in ordered_topics:
            query = ' '.join(word for word in topic)
            qvec.append(query)
    
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
    
        sims_ = []
        for i in range(len(qvec)):
            topic_sims = []
            best_sims = []
            a = vectorizer.transform([qvec[i]])
            for j in range(X.shape[0]):
                b = X[j]
                topic_sims.append((j, cosine_similarity(a, b)[0][0], len(sentences[j])))
            topic_sims = sorted(topic_sims, key=lambda x: x[1], reverse=True)
            sims_.append(topic_sims)
    
        best_sents = get_best_sents(sims_, quantile)
    
        best_combo = get_best_combo(best_sents,sims_,num_of_char)
    
        summar = create_summary(best_combo,sentences)
    
        # x.loc[index,'NMF 1 sent'] = nmf_sents[0]
        x.loc[index, 'NMF summary'] = row['content']
        x.loc[index, 'NMF summary'] = summar
        
        #perform TextRank############################################ 
        
        #getting 3 best sentences
        
        #percentage of total sentences to include in summary
        perc=10/len(split_sentences(article))
        
        #summarize
        tr_sum=summarize(article, ratio=perc)
        
        #get list of <num_of_charr long best sentences
        l=[]
        for i in split_sentences(tr_sum):
            if len(i) <= num_of_char:
                l.append(i)
    #    [print(len(x)) for x in l]
        
        #get best combo of max num_of_char
        fl=[]
        total_l=0
        for i in l:
            if total_l + len(i)<= num_of_char:
                total_l+=len(i)
                fl.append(i)
                
        #Concat final combo
        TR_Summary=' '.join(fl)
            
        x.loc[index, 'TR summary'] = TR_Summary
       
    
    ################# CLUSTERING APPRAOCH ##########################################
    # =============================================================================
    # =============================================================================
    
    ## New Summary approach with clustering (good idea to switch kMeans for DBSCAN or Maximization Expectation)
        list_of_sent = split_into_sentences(article)
        list_of_sent_of_word = [process_raw(sentence) for sentence in list_of_sent]
        flatten = [word for i in list_of_sent_of_word for word in i]   
        
        
        # sentence as vector representation of average of vector representation of words
        text2vec = []
        for sentence in list_of_sent_of_word:
            sentences2vec = []
            for word in sentence:
                try:
                    sentences2vec.append(model[word])
                except:
                    pass
            sentences2vec = np.average(sentences2vec, axis=0)
            text2vec.append(sentences2vec)
                    
        
        # centroidy calego tekstu
        model_vec = []
        for word in flatten:
            try:
                model_vec.append(model[word])
            except:
                pass
     
    ###############################################################################           
    # creating centroids for article with clustering techniques!!            
    
    ### Scale vector representatino
    #from sklearn.preprocessing import StandardScaler
    #scaler=StandardScaler()
    #df_company = scaler.fit_transform(df_company)
    
    ### Perform PCA to reduce dimension to make clustering easier!!!!
                
    #from sklearn.decomposition import PCA
    #pca = PCA(n_components = None)
    #model_vec = pca.fit_transform(model_vec)
    #ex_var = pca.explained_variance_ratio_
    #
    #pca = PCA(n_components = 12)
    #model_vec = pca.fit_transform(model_vec)
    
        import matplotlib.pyplot as plt
        
        from sklearn.cluster import KMeans
    
    #wcss=[]
    #for i in range(1,16):
    #    kmeans=KMeans(n_clusters=i)
    #    kmeans.fit(model_vec)
    #    wcss.append(kmeans.inertia_)
    #    #inertia = Sum of squared distances of samples to their closest cluster center.
    #
    #plt.plot(range(1,16), wcss, color='green')
    #plt.title('Elbow Graph')
    #plt.xlabel('Number of clusters K')
    #plt.ylabel('WCSS')
    #plt.show()
    #Training the model
        n_clusters=3
        kmeans=KMeans(n_clusters=n_clusters)
        kmeans.fit(model_vec)
        centroids = kmeans.cluster_centers_
        cent_names = ['c1', 'c2', 'c3']
        
        for i in range(n_clusters):
            tpl = zip(cent_names, kmeans.cluster_centers_[i])
        ###########################################################################
        
        df_similiarities = pd.DataFrame(columns = cent_names)
        # indexes to ilosc zdan w tekscie czyli wiersze
        
        for i in range(n_clusters):
            a = 0
            for sentence in text2vec:
                df_similiarities.loc[a,cent_names[i]] = 1 - spatial.distance.cosine(kmeans.cluster_centers_[i], sentence)
                a+=1
        # get similiarity for each sentence vs centroid 
        df_similiarities = df_similiarities.fillna(-1)    
        
        sims_ = []
        for i in range(len(df_similiarities.columns)):
            topic_sims = []
            best_sims = []
            for j in range(len(list_of_sent)):
                topic_sims.append((j, df_similiarities.iloc[j, i], len(list_of_sent[j])))
            topic_sims = sorted(topic_sims, key=lambda x: x[1], reverse=True)
            sims_.append(topic_sims)
    
        best_sents = get_best_sents(sims_, quantile)
    
        best_combo = get_best_combo(best_sents,sims_,num_of_char)
    
        summar = create_summary(best_combo,list_of_sent)
        
        x.loc[index, 'kMeans summary'] = summar
        
    
    
    # =============================================================================
    # =============================================================================
    # =============================================================================
    # a = list of word2vec vectors, b = word2vec vectors as np arrays, c = average of the vectors 
    # one row per title and each summary
    df_sum = pd.DataFrame(columns = ['a', 'b', 'c'])
    ix_of_df = [1, 3, 4, 5, 6, 7]
    all_words = []
    
    performance=pd.DataFrame(columns=['LSI', 'LDA', 'NMF', 'TextRank', 'kMeans'])
    
    rr=0
    for article in x.iterrows():
        print('NEW_ONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',rr)
        rr+=1
        for index in ix_of_df:
    #        print(article[1][index])
            # process title and all summaries
            model_raw  = process_raw(article[1][index])
            # create word2vec for each processed model
            model_vec = []
            for word in model_raw:
                try:
                    
                    # create word2vec based on Google or own model
                    model_vec.append(model[word])
                    all_words.append((word, model[word]))
                    # if word is not in word2vec model, we skip it (it is not represented in the vector)
                except:
                    pass
                # transform to array
                model_vec_rep = np.array(model_vec)   
                # take average of the word vectors for each method
                model_vec_rep_avg = np.average(model_vec_rep, axis=0)
                
            df_sum.loc[index,['a', 'b', 'c']] = [model_vec, model_vec_rep, model_vec_rep_avg]
            # Title to title comparison = 1!
            df_sum.loc[1,'sim'] = 1
            # Calculate cosine similarity between title vector and average vector of each method
            df_sum.loc[index,['sim']] = 1 - spatial.distance.cosine(df_sum.loc[1,'c'], model_vec_rep_avg)
            # We assign 0 as similarity for the title itself, so we don't pick title as most similar
            df_sum.loc[1,'sim'] = 0
#            print(x.iloc[0,df_sum['sim'].idxmax()])
    
    #    print(x.iloc[0,df_sum['sim'].idxmax()+1]) uncomment if you want 3sent summary
            
    #    tweet = 'some words i just want to share with rest of the world'
    #    make_tweet(x.iloc[0,df_sum['sim'].idxmax()])
    
            tweet = x.iloc[0,df_sum['sim'].idxmax()]
            
            return tweet
        
#summarizor('https://www.forbes.com/sites/johnwelsheurope/2019/04/03/5-pieces-of-advice-for-ceos-in-europe-and-the-americas-from-70-of-their-peers-in-asia/#61a7b97975a6')
