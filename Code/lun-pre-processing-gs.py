# encoding: utf-8
# run with python automatic-script-vasconcellos.py >> vasconcellos-out.txt

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import os.path

from fuzzywuzzy import process
from pyscopus import Scopus

def scopus_search(string, filenumber, save_path, title):
    """Run the search string returned by the function
        string_formulation() in the digital library
        Scopus via pyscopus.

    Args:
        string: Search string to be used.

    Returns:
        search_df: Structure containing all search
            results in Scopus digital library
    """
    results = 50
    # key = '7f59af901d2d86f78a1fd60c1bf9426a' # Original
    key = '56c667e47c588caa7949591daf39d8a0'
    scopus = Scopus(key)

    try:
        search_df = scopus.search(string, count=results, view='STANDARD', type_=1)
        # print(search_df)
    except Exception as e:
        print ("Exception: " + str(e))
        return -1

    pd.options.display.max_rows = 2000
    pd.options.display.max_columns = 500
    pd.options.display.max_colwidth = 5000

    scopus_id = list(search_df['scopus_id'])
    scopus_title = list(search_df['title'])

    filename = os.path.join(save_path, "%i.txt" % filenumber)

    best_id = ''
    best_title = ''

    if scopus_id == []:
        with open(filename, "w") as gs_file:
            gs_file.write('...')
        print("Don't have the article %i in Scopus.\n" % filenumber)
        return

    if (len(scopus_id) > 1):
        ratio = process.extractOne(title, scopus_title)
        best_title = ratio[0]

        for i, j in enumerate(scopus_title):
            if j == best_title:
                best_id = scopus_id[i]
    else:
        best_id = scopus_id[0]
        best_title = scopus_title[0]

    try:
        abstract_key_df = scopus.retrieve_abstract(best_id)
        scopus_abstract = abstract_key_df['abstract']
        # print(scopus_abstract)

    except Exception as e:
        print ("Exception: " + str(e))
        scopus_abstract = 0

    print("Scopus ID: " + best_id + "\n")
    print("Title: " + best_title + "\n")
    print("Abstract: " + scopus_abstract.encode('utf-8') + "\n")

    with open(filename, "w") as gs_file:
        gs_file.write(best_title + '. ' + scopus_abstract + ' ')

    gs_file.close()

    print("\n")


def main():
    """Main function."""

    author = 'lun'

    save_path = '/exp/leonardo/Files-QGS/revisao-%s-1/GS-txt/metadata/txt/' % author


    with open('/exp/leonardo/Files-QGS/revisao-%s-1/GS.csv' % author, mode='r') as gs:

        # Skipping the GS.csv line written 'title'
        next(gs)

        # Creating a list where each element is the name of a GS article, without spaces, capital letters and '-'
        title_list = [line.replace('\n', '') for line in gs]
        # print("Compact Title List: " + str(title_list))

    gs.close()

    # string = "TITLE(\"Safe and Secure Networked Control Systems under Denial-of-Service Attacks\")"
    # string = "TITLE(\"State of the art of cyber-physical systems security: An automatic control perspective\")"

    for i in range(0, len(title_list)):
        string = "TITLE(\""
        string += title_list[i]
        string += "\")"

        filenumber = i + 1

        title = title_list[i]

        scopus_search(string, filenumber, save_path, title)

if __name__ == "__main__":
    main()
