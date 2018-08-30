import numpy
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

tokenizer = RegexpTokenizer(r'\w+')
enStop = get_stop_words('en')

# Abrindo os arquivos de texto
QGS1 = open('/home/ubuntu/workspace/PreText2/pretext/texts_Maid/QGS I.txt', 'r')
QGS2 = open('/home/ubuntu/workspace/PreText2/pretext/texts_Maid/QGS II.txt', 'r')
QGS3 = open('/home/ubuntu/workspace/PreText2/pretext/texts_Maid/QGS III.txt', 'r')
QGS4 = open('/home/ubuntu/workspace/PreText2/pretext/texts_Maid/QGS IV.txt', 'r')
QGS5 = open('/home/ubuntu/workspace/PreText2/pretext/texts_Maid/QGS V.txt', 'r')
QGS6 = open('/home/ubuntu/workspace/PreText2/pretext/texts_Maid/QGS VI.txt', 'r')
QGS7 = open('/home/ubuntu/workspace/PreText2/pretext/texts_Maid/QGS VII.txt', 'r')
QGS8 = open('/home/ubuntu/workspace/PreText2/pretext/texts_Maid/QGS VIII.txt', 'r')
QGS9 = open('/home/ubuntu/workspace/PreText2/pretext/texts_Maid/QGS IX.txt', 'r')
QGS10 = open('/home/ubuntu/workspace/PreText2/pretext/texts_Maid/QGS X.txt', 'r')

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

texts = []

# loop through document list
for i in docSet:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    #print tokens
    
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in enStop]
    
    # add tokens to list
    texts.append(stopped_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=4, num_words=4))

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
