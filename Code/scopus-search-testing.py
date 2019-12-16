# encoding: utf-8
# run with python automatic-script-vasconcellos.py >> vasconcellos-out.txt

import pandas as pd

import time

import pybliometrics
from pyscopus import Scopus
from pybliometrics.scopus import AbstractRetrieval, AuthorRetrieval, ContentAffiliationRetrieval


def scopus_search(string):
    """Run the search string returned by the function
        string_formulation() in the digital library
        Scopus via pyscopus.

    Args:
        string: Search string to be used.

    Returns:
        search_df: Structure containing all search
            results in Scopus digital library
    """
    results = 2000
    key = '7f59af901d2d86f78a1fd60c1bf9426a'
    scopus = Scopus(key)

    try:
        search_df = scopus.search(string, count=results, view='STANDARD', type_=1)
        # print("number of results without improvement:", len(search_df))
        # print(search_df)
    except Exception as e:
        print ("Exception: " + str(e))
        return -1

    pd.options.display.max_rows = 2000
    pd.options.display.max_columns = 500
    pd.options.display.max_colwidth = 5000

    scopus_ids_list = list(search_df['scopus_id'])
    scopus_titles_list = list(search_df['title'])

    # print(search_df[['scopus_id']])

    scopus_abstract_list = []
    scopus_keywords_list = []

    # pub_info = scopus.retrieve_abstract('85058033875')
    # print(pub_info)

    pybliometrics.utils.create_config( )

    ab = AbstractRetrieval("10.1016/j.softx.2019.100263")
    print(ab.title)

    for i in range(0, len(scopus_ids_list)):
        try:
            abstract_key_df = scopus.retrieve_abstract(scopus_ids_list[i])
            scopus_abstract_list.append(abstract_key_df['abstract'])
            scopus_keywords_list.append(abstract_key_df.keys())
        except Exception as e:
            print ("Exception: " + str(e))
            scopus_abstract_list.append(0)
            scopus_keywords_list.append(0)




    print(scopus_abstract_list)



    print(scopus_ids_list, scopus_titles_list)

    #search_df[['title']].to_csv("/home/fuchs/Documentos/MESTRADO/Masters/Code/Exits/Result.csv", index_label=False,
    #                            encoding='utf-8', index=False, header=True, sep='\t')

    return int(len(search_df))


def main():
    """Main function."""

    start_time = time.time()

    # 807 results
    #string = "TITLE-ABS-KEY((\"risk\" AND \"software\" AND \"management\" AND \"project\" AND \"risk management\" AND \"risks\" AND \"projects\" AND \"development\")) AND PUBYEAR < 2017"

    # 3146 results
    #string = "TITLE-ABS-KEY(((\"risk\" OR \"systems\") AND (\"software\" OR \"environmental\") AND (\"management\" OR \"assessment\") AND (\"project\" OR \"systems\") AND (\"risk management\") AND (\"risks\" OR \"challenges\") AND (\"projects\" OR \"systems\") AND (\"development\" OR \"failure\"))) AND PUBYEAR < 2017"

    # 5000 results
    # string = "TITLE-ABS-KEY(((\"risk\" OR \"systems\") AND (\"software\" OR \"environmental\") AND (\"management\" OR \"assessment\") AND (\"project\" OR \"systems\") AND (\"risk management\"))) AND PUBYEAR < 2017"

    string = "TITLE(\"Safe and Secure Networked Control Systems under Denial-of-Service Attacks\")"

    scopus_search(string)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
