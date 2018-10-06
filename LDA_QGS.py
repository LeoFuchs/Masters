from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import load_files
import matplotlib.pyplot as plt
import math
import wordcloud

number_topics = 2
number_words = 10
max_document_frequency = 1.0
min_document_frequency = 0.4
ngram = (1, 3)
max_features = None

alpha = None
beta = None
learning = 'batch'  # Bacth or Online
iterations = 500


# Print the topics with the words in order
def print_top_words(model, feature_names, number_words):
    for topic_index, topic in enumerate(model.components_):
        message = "Topic %d: " % (topic_index + 1)
        message += "{"
        message += " - ".join([feature_names[i]
                               for i in topic.argsort()[:-number_words - 1:-1]])
        message += "}\n"
        print(message)


# Load training dataset
files = load_files(container_path='/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/QGS-txt', encoding="iso-8859-1")

# Extract words and vectorize dataset
tf_vectorizer = CountVectorizer(max_df=max_document_frequency,
                                min_df=min_document_frequency,
                                ngram_range=ngram,
                                max_features=max_features,
                                stop_words='english')

tf = tf_vectorizer.fit_transform(files.data)

# Save word names in a dicionary
dic = tf_vectorizer.get_feature_names()
print(dic)


# Execute lda and training
lda = LatentDirichletAllocation(n_components=number_topics,
                                doc_topic_prior=alpha,
                                topic_word_prior=beta,
                                learning_method=learning,
                                learning_decay=0.7,
                                learning_offset=10.0,
                                max_iter=iterations,
                                batch_size=128,
                                evaluate_every=-1,
                                total_samples=1000000.0,
                                perp_tol=0.1,
                                mean_change_tol=0.001,
                                max_doc_update_iter=100,
                                random_state=None)
lda.fit(tf)

# Print the topics (number_topics) with the words (number_words)
print_top_words(lda, dic, number_words)

# Print the wordcloud

for i in range(0, number_topics - 1):
    termsInTopic = lda.components_[i].argsort()[:-number_words - 1:-1]
    termsAndCounts = []
    for term in termsInTopic:
        termsAndCounts.append((str(dic[term].encode('utf-8').strip()), math.ceil(lda.components_[i][term] * 1000)))
    cloud = wordcloud.WordCloud(background_color="white")
    cloud.generate_from_frequencies(dict(termsAndCounts))
    plt.imshow(cloud)
    plt.axis("off")
    plt.savefig(str(i))
    plt.show()

