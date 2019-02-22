from pyscopus import Scopus
import pandas as pd
import re

#Number of results returned
results = 5000

#Key acess of Scopus
key = '56c667e47c588caa7949591daf39d8a0'
scopus = Scopus(key)

string = raw_input("Digite a string desejada:\n")
#string = re.sub(r'(")', r'\"', string)

#Executing SCOPUS search with pyscopus
search_df = scopus.search(string, count = results)
print("Number of Results:", len(search_df))

#Editing the parameters of the returned dataframe
# https://pandas.pydata.org/pandas-docs/stable/options.html
pd.options.display.max_rows = 99999
pd.options.display.max_colwidth = 250

# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html

search_df[['title']].to_csv("Resultado.csv", index_label = False, encoding = 'utf-8', index = False, header = True, sep = '\t')