# encoding: utf-8
from __future__ import print_function
from time import time
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups

number_topics = 2
number_words = 5
max_document_frequency = 1.0
min_document_frequency = 0.4
ngram = (1, 3)
max_features = None

iterations = 5000

# Imprime os tópicos com as palavras em ordem
def print_top_words(model, feature_names, number_words):
    for topic_index, topic in enumerate(model.components_):
        message = "Topic %d: " % (topic_index + 1)
        message += "{"
        message += " - ".join([feature_names[i]
                               for i in topic.argsort()[:-number_words - 1:-1]])
        message += "}\n"
        print(message)

# Carrega o dataset de treinamento
files = load_files(container_path = '/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/QGS-ia-txt/metadata', encoding="iso-8859-1")

# Usa tf-idf para o NMF.
tfidf_vectorizer = TfidfVectorizer(max_df = max_document_frequency,
                                   min_df = min_document_frequency,
                                   ngram_range= ngram,
                                   max_features = max_features,
                                   stop_words='english')

tfidf = tfidf_vectorizer.fit_transform(files.data)

# Salva os nomes das palavras em um dicionário
dic = tfidf_vectorizer.get_feature_names()

# SVD para reduzir a dimensionalidade
svd_model = TruncatedSVD(n_components = number_topics,
                         algorithm = 'randomized',
                         n_iter = iterations)

# Pipeline do tf-idf + SVD, fit e aplicando nos arquivos
svd_transformer = Pipeline([('tfidf', tfidf_vectorizer), ('svd', svd_model)])

svd_transformer.fit_transform(files.data)

# Imprime os (number_topics) tópicos com as (number_words) palavras
print_top_words(svd_model, dic, number_words)
