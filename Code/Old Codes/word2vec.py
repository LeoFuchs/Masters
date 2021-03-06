# encoding: utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import load_files
import gensim
import Levenshtein
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

number_topics = 2
number_words = 8

word2vec_selected_words = 1
word2vec_total_words = 30

levenshtein_distance = 4
n_gram = (1, 3)

max_document_frequency = 1.0
min_document_frequency = 0.4
max_features = None

alpha = None
beta = None
learning = 'batch'  # Batch ou Online
iterations = 5000

# Where is the best?
porter = PorterStemmer()
lancaster = LancasterStemmer()

# Imprime os tópicos com as palavras em ordem
def print_top_words(model, feature_names, number_words):
    for topic_index, topic in enumerate(model.components_):
        message = "Topic %d: " % (topic_index + 1)
        message += "{"
        message += " - ".join([feature_names[i]
                               for i in topic.argsort()[:-number_words - 1:-1]])
        message += "}\n"
        print(message)

# Imprime a string sem a melhoria do word2vec
def print_string_no_improvement(model, feature_names, number_words):

  print("String without improvement:")

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

# Imprime a string com a melhoria do word2vec
def print_string_with_improvement(model, feature_names, number_words):

  print("String with improvement:")

  wiki = gensim.models.KeyedVectors.load_word2vec_format('/home/fuchs/Documentos/MESTRADO/Datasets/wiki-news-300d-1M.vec')

  message = ("TITLE-ABS-KEY(")

  for topic_index, topic in enumerate(model.components_):

    counter = 0

    message += "("

    for i in topic.argsort()[:-number_words - 1:-1]:

        counter = counter + 1

        #teste = [('systematic', 0.7266719341278076), ('mining', 0.7077047228813171), ('dataset', 0.6963621377944946), ('statistics', 0.6708579659461975), ('data--', 0.6617765426635742)]
        #teste = [i[0] for i in teste]

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
            message += "\" OR \"".join(final_similar_word[m] for m in range(0, word2vec_selected_words)) #Where defined the number of similar words

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
print("The %d topics with your %d words in textual format: " % (number_topics, number_words))

print_top_words(lda, dic, number_words)

print_string_no_improvement(lda, dic, number_words)

print("\n")

print_string_with_improvement(lda, dic, number_words)

#model = gensim.models.KeyedVectors.load_word2vec_format('/home/fuchs/Documentos/MESTRADO/Datasets/wiki-news-300d-1M.vec')
#similar = model.most_similar(positive=['man'], topn = 5)