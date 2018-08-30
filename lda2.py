from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# Abrindo os arquivos de texto
QGS1 = open('/home/fuchs/Documentos/Mestrado/texts_Maid/QGS I.txt', 'r')
QGS2 = open('/home/fuchs/Documentos/Mestrado/texts_Maid/QGS II.txt', 'r')
QGS3 = open('/home/fuchs/Documentos/Mestrado/texts_Maid/QGS III.txt', 'r')
QGS4 = open('/home/fuchs/Documentos/Mestrado/texts_Maid/QGS IV.txt', 'r')
QGS5 = open('/home/fuchs/Documentos/Mestrado/texts_Maid/QGS V.txt', 'r')
QGS6 = open('/home/fuchs/Documentos/Mestrado/texts_Maid/QGS VI.txt', 'r')
QGS7 = open('/home/fuchs/Documentos/Mestrado/texts_Maid/QGS VII.txt', 'r')
QGS8 = open('/home/fuchs/Documentos/Mestrado/texts_Maid/QGS VIII.txt', 'r')
QGS9 = open('/home/fuchs/Documentos/Mestrado/texts_Maid/QGS IX.txt', 'r')
QGS10 = open('/home/fuchs/Documentos/Mestrado/texts_Maid/QGS X.txt', 'r')

# Criando as listas com o conteudo dos arquivos
#listQGS1 = []
textQGS1 = QGS1.read()
#listQGS1.append(textQGS1)

#listQGS2 = []
textQGS2 = QGS2.read()
#listQGS2.append(textQGS2)

#listQGS3 = []
textQGS3 = QGS3.read()
#listQGS3.append(textQGS3)

#listQGS4 = []
textQGS4 = QGS4.read()
#listQGS4.append(textQGS4)

#listQGS5 = []
textQGS5 = QGS5.read()
#listQGS5.append(textQGS5)

#listQGS6 = []
textQGS6 = QGS6.read()
#listQGS6.append(textQGS6)

#listQGS7 = []
textQGS7 = QGS7.read()
#listQGS7.append(textQGS7)

#listQGS8 = []
textQGS8 = QGS8.read()
#listQGS8.append(textQGS8)

#listQGS9 = []
textQGS9 = QGS9.read()
#listQGS9.append(textQGS9)

#listQGS10 = []
textQGS10 = QGS10.read()
#listQGS10.append(textQGS10)

# Aplicando o LDA
docSet = [textQGS1, textQGS2, textQGS3, textQGS4, textQGS5, textQGS6, textQGS7, textQGS8, textQGS9, textQGS10]

docClean = [clean(doc).split() for doc in docSet] 

dictionary = corpora.Dictionary(docClean)
docTermMatrix = [dictionary.doc2bow(doc) for doc in docClean]

Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(docTermMatrix, num_topics=3, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=3, num_words=3))

# Fechando os arquivos
QGS1.close()
QGS2.close()
QGS3.close()
QGS4.close()
QGS5.close()
QGS6.close()
QGS7.close()
QGS8.close()
QGS9.close()
QGS10.close()
