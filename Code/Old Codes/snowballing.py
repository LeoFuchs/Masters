# encoding: utf-8
import graphviz
import numpy as np
import os
from graphviz import Graph
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    #Rodando o CERMINE
    cermine = "java -cp cermine-impl-1.13-jar-with-dependencies.jar pl.edu.icm.cermine.ContentExtractor -path /home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-vasconcellos/GS-pdf/ -outputs zones"
    returned = os.system(cermine)  # returns the exit code in unix
    print(returned)

    with open('/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-vasconcellos/GS.csv', mode = 'r') as GS:

        #Pulando a linha do GS.csv escrita 'title'
        next(GS)

        #Criando uma lista onde cada elemento é o nome de um artigo do GS, sem espaços, minusculos e sem -
        title_list = [line.strip().lower().replace(' ', '').replace('-', '') for line in GS]

        #Criando uma lista auxiliar de tamanho n no formato [1, 2, 3, 4, 5, ... , n]
        list = range(1, len(title_list) + 1)

    GS.close()

    #Inicializando o Grafo com os seus respectivos nós
    g = Graph('Snowballing Graph', strict = True)
    for i in list:
        g.node('%02d' % i, shape = 'circle')

    #Analisando a citação de cada um dos artigos
    for i in range(1, len(title_list) + 1):
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

                for j in range(1, len(title_list) + 1):
                    if i != j:
                        if title_list[j-1] in reader:
                            #print("O artigo GS-%02.d cita o artigo %02.d.\n" % (i, j))
                            g.edge('%02d' % i, '%02d' % j)
                            g.edge('%02d' % j, '%02d' % i)

                    #train_group = [title_list[j-1], reader]
                    #tfidf_vectorizer = TfidfVectorizer()
                    #tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_group)
                    #matSimilaridade = cosine_similarity(tfidf_matrix_train[0], tfidf_matrix_train[1])

        file.close()

    results_list = [1, 2, 3, 4]
    for i in results_list:
        g.node('%02d' % i, shape = 'circle', color = 'red')


    r = graphviz.Source(g, filename="snowballing_graph", directory='/home/fuchs/Documentos/MESTRADO/Masters/Code/Exits/Snowballing/', format="ps")
    #r.render()
    r.view()

if __name__ == "__main__":
    main()



