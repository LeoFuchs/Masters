from pyscopus import Scopus

key = '56c667e47c588caa7949591daf39d8a0'

scopus = Scopus(key)

string = "TITLE-ABS-KEY(((\"software process improvement\") AND (\"business goal\" OR \"strategic\" OR \"goal oriented\" OR \"business oriented\" OR \"business strategy\") AND (\"alignment\" OR \"in line with\" OR \"geared to\" OR \"aligned with\" OR \"linking\") AND (\"method\" OR \"approach\" OR \"framework\" OR \"methodology\")))"

search_df = scopus.search(string, count = 30)

print(search_df)

#print(search_df[['title']])