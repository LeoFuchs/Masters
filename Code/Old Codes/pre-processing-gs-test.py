# encoding: utf-8
# run with python automatic-script-vasconcellos.py >> vasconcellos-out.txt

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import os.path

from fuzzywuzzy import process
from pyscopus import Scopus

def scopus_search(string, title):
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

    print(scopus_id, scopus_title)

    if scopus_id == []:
        with open(filename, "w") as gs_file:
            gs_file.write('...')
        print("Don't have the article %i in Scopus.\n" % filenumber)
        return

    if (len(scopus_id) > 1):
        ratio = process.extractOne(title, scopus_title)
        best_title = ratio[0]

    print(ratio)
    best_id = ''

    for i, j in enumerate(scopus_title):
        if j == best_title:
            best_id = scopus_id[i]

    print("Best Title: " + str(best_title) + " Best ID: " + str(best_id))

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

    print("\n")


def main():
    """Main function."""

    # string = "TITLE(\"Safe and Secure Networked Control Systems under Denial-of-Service Attacks\")"
    # string = "TITLE(\"State of the art of cyber-physical systems security: An automatic control perspective\")"
    # string = "TITLE(\"Data framing attack on state estimation\")"
    string = "TITLE(\"Smart grid data integrity attacks\")"


    title = "Smart grid data integrity attacks"

    scopus_search(string, title)

if __name__ == "__main__":
    main()
