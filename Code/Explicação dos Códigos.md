LDA-QGS.py: A partir dos arquivos do QGS, roda o LDA a partir de um dado valor de palavras e tópicos.

LSA-QGS.py: A partir dos arquivos do QGS, roda o LSA a partir de um dado valor de palavras e tópicos.

NMF-QGS.py: A partir dos arquivos do QGS, roda o NMF a partir de um dado valor de palavras e tópicos.

Scopus.py: A partir de uma string digitada no terminal, se obtém o resultado da busca desta string no Scopus, salva no arquivo Resultado.csv

Similaridade.py: A partir de dois arquivos de entrada (Resultado.csv = resultado da busca realizada por Scopus.py e QGS.csv = possuindo o nome de todos os artigos do QGS) se gera o arquivo Saida.csv contendo o nome dos artigos do QGS.csv em comparação de similaridade com os artigos do Resultado.csv.

word2vec.py: A ideia é que a partir de um certo número de tópicos e palavras, roda-se o LDA fornecendo um conjunto de palavras. A partir deste conjunto de palavras são geradas as palavras mais semelhantes a elas utilizando o word2vec, e em seguida é formulada uma string com refinamento onde está incluida além das palavras do LDA as palavras obtidas no Word2Vec.
