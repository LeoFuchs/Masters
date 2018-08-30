from tabulate import tabulate
import csv
import os

# Criando o Header da Tabela
listHeader = []
header = open('discover.names', 'r')
text = header.readlines()

for line in text :
    listHeader.append(line)
    
listHeader.pop(0)    #Removendo primeiro elemento

listHeader = [item.partition(":")[0] for item in listHeader] #Removendo sufixo
header.close()

# Criando o conteudo da Tabela
listContent = []
with open('discover.data', 'r') as csvfile:
    aux = csv.reader(csvfile, delimiter=',')
    
    for line in aux:
        listContent.append(line)

csvfile.close()

# Criando a Tabela completa
tabulate = tabulate(listContent, headers=listHeader)

# Salvando a Tabela no arquivo table.txt
os.remove("table.txt")

with open("table.txt", "a") as table:
    table.write(tabulate)
    
table.close()