# encoding: utf-8
import gensim
import Levenshtein
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

from pyscopus import Scopus


# Gera a representação de bag-of-words do QGS
def bag_of_words():

    n_gram = (1, 3)

    max_document_frequency = 1.0
    min_document_frequency = 0.3
    max_features = None

    # Carrega o dataset de treinamento
    files = load_files(container_path = '/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/QGS-ia-txt/metadata', encoding="iso-8859-1")

    # Extrai as palavras e vetoriza o dataset
    tf_vectorizer = CountVectorizer(max_df = max_document_frequency,
                                    min_df = min_document_frequency,
                                    ngram_range = n_gram,
                                    max_features = max_features,
                                    stop_words = 'english')

    tf = tf_vectorizer.fit_transform(files.data)

    # Salva os nomes das palavras em um dicionário
    dic = tf_vectorizer.get_feature_names()

    return dic, tf

# Executa o algoritmo LDA na representação de bag-of-words
def lda_algorithm(tf, lda_iterations):

    alpha = None
    beta = None
    learning = 'batch'  # Batch ou Online

    # Executa o LDA e treina-o
    lda = LatentDirichletAllocation(n_components = number_topics,
                                    doc_topic_prior = alpha,
                                    topic_word_prior = beta,
                                    learning_method = learning,
                                    learning_decay = 0.7,
                                    learning_offset = 10.0,
                                    max_iter = lda_iterations,
                                    batch_size = 128,
                                    evaluate_every = -1,
                                    total_samples = 1000000.0,
                                    perp_tol = 0.1,
                                    mean_change_tol = 0.001,
                                    max_doc_update_iter = 100,
                                    random_state = None)

    lda.fit(tf)

    return lda

# Imprime os (number_topics) tópicos com as suas respectivas (number_words) palavras
def print_top_words(model, feature_names, number_words, number_topics):

    print("The %d topics with your %d words in textual format: <br>" % (number_topics, number_words))

    for topic_index, topic in enumerate(model.components_):
        message = "Topic %d: " % (topic_index + 1)
        message += "{"
        message += " - ".join([feature_names[i] for i in topic.argsort()[:-number_words - 1:-1]])
        message += "} <br>"
        print(message)
    print("\n")

# Imprime a string sem a melhoria do word2vec
def print_string_no_improvement(model, feature_names, number_words, number_topics):

  print("String without improvement: <br>")

  message = ("TITLE-ABS-KEY(")

  for topic_index, topic in enumerate(model.components_):

    message += "(\""
    message += "\" AND \"".join([feature_names[i] for i in topic.argsort()[:-number_words - 1:-1]])
    message += "\")"

    if topic_index < number_topics - 1:
      message += " OR "
    else:
      message += ""

  message += ")"

  print(message)
  return message

# Imprime a string com a melhoria do word2vec
def print_string_with_improvement(model, feature_names, number_words, number_topics, similar_words, levenshtein_distance):

  print("String with improvement: <br>")

  porter = PorterStemmer()
  lancaster = LancasterStemmer()

  word2vec_total_words = 30

  wiki = gensim.models.KeyedVectors.load_word2vec_format('/home/fuchs/Documentos/MESTRADO/Datasets/wiki-news-300d-1M.vec')

  message = ("TITLE-ABS-KEY(")

  for topic_index, topic in enumerate(model.components_):

    counter = 0
    message += "("

    for i in topic.argsort()[:-number_words - 1:-1]:

        counter = counter + 1

        if " " not in feature_names[i]:
            similar_word = wiki.most_similar(positive = feature_names[i], topn = word2vec_total_words)
            similar_word = [j[0] for j in similar_word]
            #print("Similar word:", similar_word)

            stem_feature_names = lancaster.stem(feature_names[i])
            #print("Stem feature names:", stem_feature_names)

            stem_similar_word = []

            final_stem_similar_word = []
            final_similar_word = []

            for j in similar_word:
                stem_similar_word.append(lancaster.stem(j))
            #print("Stem Similar Word:", stem_similar_word)

            for number, word in enumerate(stem_similar_word):
                if stem_feature_names != word and Levenshtein.distance(stem_feature_names, word) > levenshtein_distance:

                    irrelevant = 0

                    for k in final_stem_similar_word:
                        if Levenshtein.distance(k, word) < levenshtein_distance:
                            irrelevant = 1

                    if irrelevant == 0:
                        final_stem_similar_word.append(word)
                        final_similar_word.append(similar_word[number])

            #print("Final Stem Similar Word:", final_stem_similar_word)
            #print("Final Similar Word:", final_similar_word)
            #print("\n\n\n")

        message += "(\""
        message += "\" - \"".join([feature_names[i]])

        if " " not in feature_names[i]:
            message += "\" OR \""
            message += "\" OR \"".join(final_similar_word[m] for m in range(0, similar_words)) #Where defined the number of similar words

        message += "\")"

        if counter < len(topic.argsort()[:-number_words - 1:-1]):
            message += " AND "
        else:
          message += ""

    message += ")"

    if topic_index < number_topics - 1:
        message += " OR "
    else:
        message += ""

  message += ")"

  print(message)
  return message

# Efetua a busca da string sem melhorias no Scopus
def scopus_without_improvement(string_no_improvement):
    results = 5000

    key = '56c667e47c588caa7949591daf39d8a0'
    scopus = Scopus(key)

    search_df = scopus.search(string_no_improvement, count=results)
    #print("Number of results without improvement:", len(search_df))

    pd.options.display.max_rows = 99999
    pd.options.display.max_colwidth = 250

    search_df[['title']].to_csv("ResultWithoutImprovement.csv", index_label = False, encoding ='utf-8', index = False, header = True, sep = '\t')

    return int(len(search_df))

# Efetua a busca da string com melhorias no Scopus
def scopus_with_improvement(string_with_improvement):
    results = 5000

    key = '56c667e47c588caa7949591daf39d8a0'
    scopus = Scopus(key)

    search_df = scopus.search(string_with_improvement, count=results)
    #print("Number of results with improvement:", len(search_df))

    pd.options.display.max_rows = 99999
    pd.options.display.max_colwidth = 250

    search_df[['title']].to_csv("ResultWithImprovement.csv", index_label = False, encoding = 'utf-8', index = False, header = True, sep = '\t')

    return int(len(search_df))

# Abre os arquivos que serão utilizados
def open_files():

    QGS = pd.read_csv('/home/fuchs/Documentos/MESTRADO/Masters/Code/QGS.csv', sep='\t')

    result_with_improvement = pd.read_csv('/home/fuchs/Documentos/MESTRADO/Masters/Code/ResultWithImprovement.csv',
                                          sep='\t')
    result_without_improvement = pd.read_csv(
        '/home/fuchs/Documentos/MESTRADO/Masters/Code/ResultWithoutImprovement.csv', sep='\t')

    manual_exit_with_improvement = open('/home/fuchs/Documentos/MESTRADO/Masters/Code/ExitWithImprovement.csv', 'w')
    manual_exit_without_improvement = open('/home/fuchs/Documentos/MESTRADO/Masters/Code/ExitWithoutImprovement.csv',
                                           'w')

    return QGS, result_with_improvement, result_without_improvement, manual_exit_with_improvement, manual_exit_without_improvement

# Efetua o calculo da similaridade para os resultados sem melhoria
def similarity_with_improvement(QGS, result_with_improvement, manual_exit_with_improvement):

    len_qgs = sum(1 for line in open('/home/fuchs/Documentos/MESTRADO/Masters/Code/QGS.csv')) - 1
    len_result = sum(1 for line in open('/home/fuchs/Documentos/MESTRADO/Masters/Code/ResultWithImprovement.csv')) - 1

    list_QGS = []
    list_result = []

    for i in range(0, len_qgs):
        list_QGS.append(QGS.iloc[i, 0])

    # print("Lista QGS:", list_QGS)
    # print("Tamanho Lista QGS:", len(list_QGS))

    for i in range(0, len_result):
        list_result.append(result_with_improvement.iloc[i, 0])

    # print("Lista Resultado:", list_result)
    # print("Tamanho Lista Resultado:", len(list_result))

    train_set = [list_QGS, list_result]
    train_set = [val for sublist in train_set for val in sublist]

    # print("Lista train_set:", train_set)
    # print("Elementos train_set", len(train_set))

    tfidf_vectorizer = TfidfVectorizer(stop_words ='english')
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)

    matSimilaridade = cosine_similarity(tfidf_matrix_train[0:len_qgs], tfidf_matrix_train[len_qgs:len_qgs + len_result])
    lin, col = matSimilaridade.shape

    counter_improvement = 0

    for i in range(0, lin):

        line = matSimilaridade[i]
        currentNearest = np.argsort(line)[-2:]  # Pega os x - 1 maiores elementos

        line_exit = 'QGS' + str(i + 1) + ':\t\t\t' + list_QGS[i] + '\t' + '\n'

        for j in range(1, len(currentNearest)):
            book = currentNearest[-j]
            line_exit = line_exit + '\t\t\t\t' + list_result[book].strip() + '\t' '\n'

        if Levenshtein.distance(list_QGS[i], list_result[book]) < 10:
            counter_improvement = counter_improvement + 1

        line_exit = line_exit + "\n"

        manual_exit_with_improvement.write(line_exit)
        manual_exit_with_improvement.flush()

    #print("Number of QGS articles founded (with improvement):", counter_improvement)

    return counter_improvement

# Efetua o calculo da similaridade para os resultados com melhoria
def similarity_wihout_improvement(QGS, result_without_improvement, manual_exit_without_improvement):

    len_qgs = sum(1 for line in open('/home/fuchs/Documentos/MESTRADO/Masters/Code/QGS.csv')) - 1

    len_result = sum(1 for line in open('/home/fuchs/Documentos/MESTRADO/Masters/Code/ResultWithoutImprovement.csv')) - 1

    list_qgs = []
    list_result = []

    for i in range(0, len_qgs):
        list_qgs.append(QGS.iloc[i, 0])

    # print("Lista QGS:", list_qgs)
    # print("Tamanho Lista QGS:", len(list_qgs))

    for i in range(0, len_result):
        list_result.append(result_without_improvement.iloc[i, 0])

    # print("Lista Resultado:", list_result)
    # print("Tamanho Lista Resultado:", len(list_result))

    train_set = [list_qgs, list_result]
    train_set = [val for sublist in train_set for val in sublist]

    # print("Lista train_set:", train_set)
    # print("Elementos train_set", len(train_set))

    tfidf_vectorizer = TfidfVectorizer(stop_words ='english')
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)

    matSimilaridade = cosine_similarity(tfidf_matrix_train[0:len_qgs], tfidf_matrix_train[len_qgs:len_qgs + len_result])
    lin, col = matSimilaridade.shape

    counter_no_improvement = 0

    for i in range(0, lin):

        line = matSimilaridade[i]
        currentNearest = np.argsort(line)[-2:]  # Pega os x - 1 maiores elementos

        line_exit = 'QGS' + str(i + 1) + ':\t\t\t' + list_qgs[i] + '\t' + '\n'

        for j in range(1, len(currentNearest)):
            book = currentNearest[-j]
            line_exit = line_exit + '\t\t\t\t' + list_result[book].strip() + '\t' '\n'

        if Levenshtein.distance(list_qgs[i], list_result[book]) < 10:
            counter_no_improvement = counter_no_improvement + 1

        line_exit = line_exit + "\n"

        manual_exit_without_improvement.write(line_exit)
        manual_exit_without_improvement.flush()

    #print("Number of QGS articles founded (without improvement):", counter_no_improvement)

    return counter_no_improvement

lda_iterations = int(input("LDA Iterations: "))
number_topics = int(input("LDA Topics: "))
number_words = int(input("LDA Words: "))

similar_words = int(input("Similar Words: "))
levenshtein_distance = int(input("Levenshtein Distance: "))
print("\n")

print("__Metadata from QGS with " + str(number_topics) + " topics and " + str(number_words) + " words. Improvement with more " + str(similar_words) + " similar words and Levenshtein distance is " + str(levenshtein_distance) + "__")
print("\n")

dic, tf = bag_of_words()
lda = lda_algorithm(tf, lda_iterations)
print_top_words(lda, dic, number_words, number_topics)

string_no_improvement = print_string_no_improvement(lda, dic, number_words, number_topics)
print("\n")
string_with_improvement = print_string_with_improvement(lda, dic, number_words, number_topics, similar_words, levenshtein_distance)

print("\n")
print("**Results from Scopus Search:**\n")

result_1 = scopus_without_improvement(string_no_improvement)
result_2 = scopus_with_improvement(string_with_improvement)

raw_input("...")

QGS, result_with_improvement, result_without_improvement, manual_exit_with_improvement, manual_exit_without_improvement = open_files()
counter_1 = similarity_wihout_improvement(QGS, result_without_improvement, manual_exit_without_improvement)
counter_2 = similarity_with_improvement(QGS, result_with_improvement, manual_exit_with_improvement)

print("<font color='red'> **String without improvement**: " + str(result_1) + " results where " + str(counter_1) + " of the 46 QGS articles are present in the search <br>")
print("**String with improvement**: " + str(result_2) + " results where " + str(counter_2) + " of the 46 QGS articles are present in the search </font>")