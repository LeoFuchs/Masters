# encoding: utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import load_files
import gensim

number_topics = 2
number_words = 5
max_document_frequency = 1.0
min_document_frequency = 0.4
ngram = (1, 2)
max_features = None

alpha = None
beta = None
learning = 'batch'  # Batch ou Online
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
def print_string_with_improvement1(model, feature_names, number_words):
  print("String with improvement:")

  wiki = gensim.models.KeyedVectors.load_word2vec_format('/home/fuchs/Documentos/MESTRADO/Datasets/wiki-news-300d-1M.vec')

  message = ("TITLE-ABS-KEY(")

  for topic_index, topic in enumerate(model.components_):

    counter = 0

    message += "("

    for i in topic.argsort()[:-number_words - 1:-1]:

        counter = counter + 1
        similar_word = wiki.most_similar(positive = feature_names[i], topn = 2)
        similar_word = [j[0] for j in similar_word]

        message += "(\""
        message += "\" - \"".join([feature_names[i]])
        message += "\" OR \""

        message += "\" OR \"".join(similar_word)
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

# Imprime a string com a melhoria do word2vec
def print_string_with_improvement2(model, feature_names, number_words):
  print("String with improvement:")

  wiki = gensim.models.KeyedVectors.load_word2vec_format('/home/fuchs/Documentos/MESTRADO/Datasets/wiki-news-300d-1M.vec')

  message = ("TITLE-ABS-KEY(")

  for topic_index, topic in enumerate(model.components_):

    counter = 0

    message += "("

    for i in topic.argsort()[:-number_words - 1:-1]:

        counter = counter + 1
        similar_word = wiki.most_similar(positive = feature_names[i], topn = 2)
        similar_word = [j[0] for j in similar_word]

        message += "(\""
        message += "\" - \"".join([feature_names[i]])
        message += "\" AND \""

        message += "\" AND \"".join(similar_word)
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
print("The %d topics with your %d words in textual format: " % (number_topics, number_words))

print_top_words(lda, dic, number_words)

print_string_no_improvement(lda, dic, number_words)

print("\n")

print_string_with_improvement1(lda, dic, number_words)

#print_string_with_improvement2(lda, dic, number_words)


#model = gensim.models.KeyedVectors.load_word2vec_format('/home/fuchs/Documentos/MESTRADO/Datasets/wiki-news-300d-1M.vec')
#similar = model.most_similar(positive=['man'], topn = 5)

#teste = [('Data', 0.7266719341278076), ('datasets', 0.7077047228813171), ('dataset', 0.6963621377944946), ('statistics', 0.6708579659461975), ('data--', 0.6617765426635742)]
#teste = [i[0] for i in teste]