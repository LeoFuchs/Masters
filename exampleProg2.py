from pyscopus import Scopus
import pandas as pd
import re

#Quantos resultados quero que retorne?
resultados = 500

#Chave de acesso Scopus
key = '56c667e47c588caa7949591daf39d8a0'

scopus = Scopus(key)

#Recebendo string de busca e formatando-a adequadamente
string = raw_input("Digite a string desejada:\n")
#string = re.sub(r'(")', r'\"', string)

#Executando a busca na SCOPUS com o pyscopus
search_df = scopus.search(string, count = resultados)

#print(search_df)

#Editando os parametros do dataframe retornado
# https://pandas.pydata.org/pandas-docs/stable/options.html
pd.options.display.max_rows = 99999
pd.options.display.max_colwidth = 250

# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html

#print(search_df[['title']])
search_df[['title']].to_csv("Resultado.csv", index_label=False, encoding = 'utf-8', index=False, header=True, sep='\t')