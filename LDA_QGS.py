from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import load_files

#n_features = 10
n_topics = 2
n_top_words = 10

# Print the n_top_words in order
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " - ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

# Training dataset
files = load_files(container_path='/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/QGS-txt', encoding="iso-8859-1")


# extract fetures and vectorize dataset
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                #max_features=n_features,
                                ngram_range=(1,3),
                                stop_words='english')
#print files.data
tf = tf_vectorizer.fit_transform(files.data)

#save features
dic = tf_vectorizer.get_feature_names()

#print dic

lda = LatentDirichletAllocation(n_components=n_topics,
                                max_iter=500,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

# train LDA
p1 = lda.fit(tf)


print_top_words(lda, dic, n_top_words)