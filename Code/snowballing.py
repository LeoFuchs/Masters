# encoding: utf-8
from graphviz import Graph
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import graphviz
import numpy as np


list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
#list = [30]



with open('/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-vasconcellos/GS.csv', mode = 'r') as GS:

    next(GS)
    title_list = [line.strip().lower().replace(' ', '').replace('-', '') for line in GS]
    #print (title_list, len(title_list))

GS.close()

for i in range(1, 31):
    article_name = '/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-vasconcellos/GS-pdf/%d.cermzones' % i
    with open('%s' % article_name, mode = 'r') as file:

        # Manipulando o arquivo de entrada

        if i != 16:
            #Tornando todas as letras minusculas
            reader = file.read().lower()

            #Removendo as quebras de linha
            reader = reader.strip().replace('\n', ' ').replace('\r', '')

            #Removendo os espaços e caracteres especiais
            reader = reader.replace(' ', '').replace('-', '')

            #Filtrando apenas a parte das referências
            sep = "<zonelabel=\"gen_references\">"
            reader = reader.split(sep, 1)[1]

            #print(reader)

            for j in range(1, 31):
                if i-1 != j-1:
                    #print(i, j)
                    #print(title_list[i-1], title_list[j-1])
                    if title_list[j-1] in reader:
                        print("O artigo GS-%02.d cita o artigo %02.d.\n" % (i, j))



        #print(title_list[i-1])
        #for j in range(1, 31):
            #if i-1 != j-1:
                #print(i,j)
                #print(title_list[i-1], title_list[j-1])

                #train_group = [title_list[j-1], reader]
                #tfidf_vectorizer = TfidfVectorizer()
                #tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_group)

                #print(tfidf_matrix_train)
                #matSimilaridade = cosine_similarity(tfidf_matrix_train[0], tfidf_matrix_train[1])
                #print(matSimilaridade[0])
        #print("\n")
    file.close()

g = Graph('Snowballing Graph', strict = True)

for i in list:
    g.attr('node', shape = 'circle')
    g.node('%02d' % i)

g.edge('02', '03')
g.edge('08', '09')
g.edge('10', '30')
g.edge('11', '10')
g.edge('14', '09')
g.edge('14', '12')
g.edge('15', '04')
g.edge('15', '18')
g.edge('16', '02')
g.edge('17', '02')
g.edge('17', '18')
g.edge('18', '04')
g.edge('18', '05')
g.edge('19', '04')
g.edge('19', '14')
g.edge('21', '02')
g.edge('21', '20')
g.edge('24', '30')
g.edge('25', '30')
g.edge('28', '22')
g.edge('29', '14')

min_df = 0.4
number_topics = 2
number_words = 10
enrichment = 1
#r = graphviz.Source(g, filename="graph-with-%0.1f-%d-%d-%d" % (min_df, number_topics, number_words, enrichment), directory='/home/fuchs/Documentos/MESTRADO/Masters/Code/Exits/Snowballing/', format="ps")

#r.render()
#r.view()



