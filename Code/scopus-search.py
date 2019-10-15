# encoding: utf-8
# run with python automatic-script-vasconcellos.py >> vasconcellos-out.txt
import torch
import Levenshtein
import csv
import pandas as pd
import numpy as np
import graphviz
import glob
import os
import shutil
import random
import sys
import time

from itertools import islice
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from fuzzywuzzy import process
from pytorch_transformers import BertTokenizer, BertForMaskedLM
from nltk.stem import LancasterStemmer
from graphviz import Graph
from pyscopus import Scopus


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
        print("number of results without improvement:", len(search_df))
    except Exception as e:
        print ("Exception: " + str(e))
        return -1

    pd.options.display.max_rows = 2000
    pd.options.display.max_colwidth = 250

    print(search_df[['title']])

    search_df[['title']].to_csv("/home/fuchs/Documentos/MESTRADO/Masters/Code/Exits/Result.csv", index_label=False,
                                encoding='utf-8', index=False, header=True, sep='\t')

    return int(len(search_df))


def main():
    """Main function."""

    start_time = time.time()

    # 807 results
    #string = "TITLE-ABS-KEY((\"risk\" AND \"software\" AND \"management\" AND \"project\" AND \"risk management\" AND \"risks\" AND \"projects\" AND \"development\")) AND PUBYEAR < 2017"

    # 3146 results
    #string = "TITLE-ABS-KEY(((\"risk\" OR \"systems\") AND (\"software\" OR \"environmental\") AND (\"management\" OR \"assessment\") AND (\"project\" OR \"systems\") AND (\"risk management\") AND (\"risks\" OR \"challenges\") AND (\"projects\" OR \"systems\") AND (\"development\" OR \"failure\"))) AND PUBYEAR < 2017"

    # 5000 results
    string = "TITLE-ABS-KEY(((\"risk\" OR \"systems\") AND (\"software\" OR \"environmental\") AND (\"management\" OR \"assessment\") AND (\"project\" OR \"systems\") AND (\"risk management\"))) AND PUBYEAR < 2017"

    scopus_search(string)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
