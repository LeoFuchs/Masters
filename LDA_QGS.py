# encoding: utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import load_files
import matplotlib.pyplot as plt
import math
import wordcloud

number_topics = 5
number_words = 2
max_document_frequency = 1.0
min_document_frequency = 0.4
ngram = (1, 3)
max_features = None

alpha = None
beta = None
learning = 'batch'  # Bacth ou Online
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
files = load_files(container_path = '/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/QGS-ia-txt/', encoding="iso-8859-1")

# Extrai as palavras e vetoriza o dataset
tf_vectorizer = CountVectorizer(max_df = max_document_frequency,
                                min_df = min_document_frequency,
                                ngram_range = ngram,
                                max_features = max_features,
                                stop_words = 'english')

tf = tf_vectorizer.fit_transform(files.data)

# Salva os nomes das palavras em um dicionário
dic = tf_vectorizer.get_feature_names()

# Executa o LDA e treina-o
lda = LatentDirichletAllocation(n_components = number_topics,
                                doc_topic_prior = alpha,
                                topic_word_prior = beta,
                                learning_method = learning,
                                learning_decay = 0.7,
                                learning_offset = 10.0,
                                max_iter = iterations,
                                batch_size = 128,
                                evaluate_every = -1,
                                total_samples = 1000000.0,
                                perp_tol = 0.1,
                                mean_change_tol = 0.001,
                                max_doc_update_iter = 100,
                                random_state = None)
lda.fit(tf)

# Imprime os (number_topics) tópicos com as (number_words) palavras
print("The %d topics with your %d words in textual format: \n" % (number_topics, number_words))
print_top_words(lda, dic, number_words)
