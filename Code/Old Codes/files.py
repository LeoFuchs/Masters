import glob

read_files = glob.glob("/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-vasconcellos/QGS-txt/metadata-enrichment/txt/*.txt")

with open("result.txt", "wb") as merge_files:
    for f in read_files:
        with open(f, "rb") as infile:
            merge_files.write(infile.read())

merge_files.close()

with open("result.txt", "r") as metadata_file:
    data = metadata_file.read().strip()
    data = data.replace('\r\n', '')
    print (data)