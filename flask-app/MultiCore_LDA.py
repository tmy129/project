#!/usr/bin/env python
# coding: utf-8

import logging, gensim
import time
import pandas as pd
import numpy as np
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import jieba.analyse
import pickle
from gensim import corpora, models
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.coherencemodel import CoherenceModel

def lds_model():
    filename="./sample_data.txt"
    jieba.load_userdict('recruit.txt')

    corpus = []
    with open(filename, 'r') as f:
        for line in f:
            token_list = jieba.lcut(line, cut_all=False)
            str_tmp = ''
            for i in range(0,len(token_list)):
                str_tmp = str_tmp + ' ' + token_list[i]
            corpus.append(str_tmp)
    f.close()

    documents=corpus

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=1, max_features=10000, stop_words='english',ngram_range=(1,2))
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    non_stoplist=set()

    for i in range(len(tf_feature_names)):
        if (not (tf_feature_names[i].isdigit())):
            non_stoplist.add(tf_feature_names[i])


    texts = [[word.upper() for word in document.split() if word in non_stoplist]
            for document in corpus]


    dictionary = corpora.Dictionary(texts)
    lda_corpus = [dictionary.doc2bow(text) for text in texts]
    return texts, dictionary, lda_corpus

def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    coherence_values = []
    model_list = []
    num_topics_list = []
    for num_topics in range(start, limit, step):
        print ("processing: "+str(num_topics))
        num_topics_list.append(num_topics)
        model = models.LdaMulticore(corpus=corpus, id2word=dictionary, alpha=5.0/num_topics, num_topics=num_topics, workers=4, passes=15, per_word_topics=True, chunksize=5000)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return num_topics_list, model_list, coherence_values

def model_monitor_comparison(start,limit,step):
    texts, dictionary, lda_corpus = lds_model()
    num_topics_list, model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=lda_corpus, texts=texts, start=start, limit=limit, step=step)

    cohe_diff = 0
    best_cnt = 0
    best_model = 0
    for m in range(0,len(model_list)-1):
        diff = coherence_values[m+1] - coherence_values[m]
        if diff > cohe_diff:
            cohe_diff = diff
            best_cnt = num_topics_list[m]
            best_model = m
    pickle.dump(CoherenceModel(model=model_list[best_model], texts=texts, dictionary=dictionary, coherence='c_v'),open('model.pkl','wb'))

    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

if __name__ == "__main__":
    model_monitor_comparison(6,30,3)