from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

#58173 - acervo 0 a 58172
#58829 - total 58173 a 58828

#bibliotecaArq  = pd.read_csv('/Users/bruno/Dropbox/Coletor Biblioteca/dados_biblioteca.txt', sep='\t')
#dadosAsArq     = pd.read_csv('/Users/bruno/Dropbox/Coletor Biblioteca/dados_disciplinas_as.txt', sep='\t')
#arqEntrada = open('/Users/bruno/Dropbox/Coletor Biblioteca/dados_todos_livros.txt', 'r')
#arqSaida   = open('/Users/bruno/Dropbox/Coletor Biblioteca/sugestoes_livros_new.txt', 'w')

bibliotecaArq  = pd.read_csv('/home/bruno/Dropbox/Coletor Biblioteca/dados_biblioteca.txt', sep='\t')
dadosAsArq     = pd.read_csv('/home/bruno/Dropbox/UFMS/Coordenacao/Novo PPC/dados_disciplinas_outros.txt', sep='\t')
arqEntrada = open('/home/bruno/Dropbox/UFMS/Coordenacao/Novo PPC/dados_todos_livros_outros.txt', 'r')
arqSaida   = open('/home/bruno/Dropbox/UFMS/Coordenacao/Novo PPC/sugestoes_livros_new.txt', 'w')

lista = []
listaQtd = []
listaInfo = []

#for i in range (0, 58829):
#    if(i < 58173):
#        listaQtd.append(str(bibliotecaArq.iloc[i, 1]))
#    else:
#        listaInfo.append(dadosAsArq.iloc[i-58173, 0] + '\t' + dadosAsArq.iloc[i-58173, 1])
#    lista.append(arqEntrada.readline())

for i in range (0, 58542):
    if(i < 58173):
        listaQtd.append(str(bibliotecaArq.iloc[i, 1]))
    else:
        listaInfo.append(dadosAsArq.iloc[i-58173, 0] + '\t' + dadosAsArq.iloc[i-58173, 1])
    lista.append(arqEntrada.readline())


vect = TfidfVectorizer(stop_words='english')
tfidf_matrix = vect.fit_transform(lista)
#print tfidf_matrix.shape

#matSimilaridade = cosine_similarity(tfidf_matrix[58173:58829], tfidf_matrix[0:58173])
matSimilaridade = cosine_similarity(tfidf_matrix[58173:58542], tfidf_matrix[0:58173])
lin, col = matSimilaridade.shape

for i in range (0, lin):
    linha = matSimilaridade[i]
    currentNearest = np.argsort(linha)[-15:]
    linhaSaida = listaInfo[i] + '\t' + lista[58173 + i] + '\n'
    for i in range(1, len(currentNearest)):
        book = currentNearest[-i]
        linhaSaida = linhaSaida + '\t\t' + lista[book].strip() + '\t' + listaQtd[book] + '\n'
    linhaSaida = linhaSaida + "\n"
    arqSaida.write(linhaSaida)
    arqSaida.flush()
    
#print tfidf_matrix.shape
#cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
