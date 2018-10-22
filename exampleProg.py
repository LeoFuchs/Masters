from elsapy.elsclient import ElsClient
from elsapy.elssearch import ElsSearch
import scopus
import json
    
## Load configuration
con_file = open("config.json")
config = json.load(con_file)
con_file.close()

## Initialize client
client = ElsClient(config['apikey'])
#client.inst_token = config['insttoken']

## Initialize doc search object and execute search, retrieving all results
doc_srch = ElsSearch('star+trek+vs+star+wars','scopus')
print(doc_srch)
doc_srch.execute(client, get_all = True)

doc_srch.hasAllResults(ElsSearch)

print("doc_srch has", len(doc_srch.results), "resultados.")

