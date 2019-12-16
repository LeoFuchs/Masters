# encoding: utf-8
# run with python automatic-script-vasconcellos.py >> vasconcellos-out.txt
import numpy as np
import sys
import pickle

from itertools import islice
from fuzzywuzzy import process


def window(seq, n):
    """Returns a sliding window (of width n) over data from the iterable

    Args:
        seq: String with the sequence of words
        n: Size of the window
    Returns:
        result: ...
    """

    it = iter(seq)
    result = tuple(islice(it, n))

    if len(result) == n:
        yield result

    for elem in it:
        result = result[1:] + (elem,)
        yield result


def snowballing():
    """Doing the snowballing of the articles presented in GS (Gold Standard).

   Args:

   Returns:
       title_list:
       adjacency_matrix:
       final_edges:
   """

    with open('/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-%s/GS.csv' % author, mode='r') as gs:

        # Skipping the GS.csv line written 'title'
        next(gs)

        # Creating a list where each element is the name of a GS article, without spaces, capital letters and '-'
        title_list = [line.strip().lower().replace(' ', '').replace('.', '') for line in gs]
        # print("Compact Title List: " + str(title_list))

    gs.close()

    adjacency_matrix = np.zeros((len(title_list), len(title_list)))
    # print(adjacency_matrix)

    final_edges = []

    # Analyzing the citations of each of the articles
    for i in range(1, len(title_list) + 1):
        article_name = '/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-%s/GS-pdf/%d.cermtxt' % (author, i)
        with open('%s' % article_name, mode='r') as file_zone:
            reader = file_zone.read().strip().lower().replace('\n', ' ').replace('\r', ''). \
                replace(' ', '').replace('.', '')
            for j in range(1, len(title_list) + 1):
                window_size = len(title_list[j - 1])
                options = ["".join(x) for x in window(reader, window_size)]

                if i != j:
                    ratio = process.extractOne(title_list[j - 1], options)
                    print("Ratio: (" + str(i) + " - " + str(j) + "): " + str(ratio))
                    if ratio is not None:
                        if ratio[1] >= 90:
                            auxiliar_list = [i, j]
                            final_edges.append(auxiliar_list)

                            adjacency_matrix[i - 1][j - 1] = 1
                            adjacency_matrix[j - 1][i - 1] = 1

            file_zone.close()

    # print ("Final edges:" + str(final_edges))
    return title_list, adjacency_matrix, final_edges


def main():
    """Main function."""

    reload(sys)
    sys.setdefaultencoding('utf-8')

    global author
    author = 'lun'

    print("Doing Snowballing...\n")
    title_list, adjacency_matrix, final_edges = snowballing()

    with open('title-list-snowballing.txt', 'wb') as title_list_snow:
        pickle.dump(title_list, title_list_snow)

    with open('adjacency-matrix-snowballing.txt', 'wb') as adjacency_matrix_snow:
        pickle.dump(adjacency_matrix, adjacency_matrix_snow)

    with open('final-edges-snowballing.txt', 'wb') as final_edges_snow:
        pickle.dump(final_edges, final_edges_snow)


if __name__ == "__main__":
    main()
