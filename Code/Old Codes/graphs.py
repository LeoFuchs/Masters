import graphviz

from graphviz import Graph
import numpy as np


def graph_snowballing(results_list):
    """It generates the graph that presenting the Gold Standard (GS)
       snowballing and which GS items were found when analyzing
       the results.

   Args:
       results_list: List with the numbers of the articles of the
           GS found, that will be used in the formulation of
           the final graph.
       min_df: The minimum number of documents that a term
           must be found in.
       number_topics: Number of topics that the Latent
           Dirichlet Allocation (LDA) algorithm should return.
       number_words: Number of similar words that will be used
           to add the search string.
       enrichment: Number of rich terms that were used in the
           search string.

   Returns:
   """

    with open('/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-vasconcellos/GS.csv', mode='r') as gs:

        # Skipping the GS.csv line written 'title'
        next(gs)

        # Creating a list where each element is the name of a GS article, without spaces, capital letters and '-'
        title_list = [line.strip().lower().replace(' ', '').replace('-', '') for line in gs]

    gs.close()

    # Creating an auxiliary list of size n in the format [1, 2, 3, 4, 5, ..., n]
    node_list = range(1, len(title_list) + 1)

    adjacency_matrix = np.zeros((len(title_list), len(title_list)))
    #print(adjacency_matrix)

    # Initializing the graph with its respective nodes
    g = Graph('Snowballing Graph', strict=True)
    for i in node_list:
        g.node('%02d' % i, shape='circle')

    # Analyzing the citations of each of the articles
    for i in range(1, len(title_list) + 1):
        article_name = '/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-vasconcellos/GS-pdf/%d.cermzones' % i
        with open('%s' % article_name, mode='r') as file_zone:

            # Manipulating the input file (REMOVE THIS I != 16 LATER)
            if i != 16:
                # Making all lowercase letters
                reader = file_zone.read().lower()

                # Removing line breaks
                reader = reader.strip().replace('\n', ' ').replace('\r', '')

                # Removing spaces and special characters
                reader = reader.replace(' ', '').replace('-', '')

                # Filtering only the part of the references in the zone file
                sep = "<zonelabel=\"gen_references\">"
                reader = reader.split(sep, 1)[1]
                # print(reader)

                for j in range(1, len(title_list) + 1):
                    if i != j:
                        if title_list[j - 1] in reader:
                            #print("the article GS-%02.d cite the article %02.d.\n" % (i, j))
                            g.edge('%02d' % i, '%02d' % j)
                            adjacency_matrix[i-1][j-1] = 1
                            adjacency_matrix[j-1][i-1] = 1
                            # g.edge('%02d' % j, '%02d' % i)

        file_zone.close()

    print(adjacency_matrix)
    final_list = []

    for z in results_list:
        final_list.append(z)
    print(final_list)

    flag = 1

    while(flag):
        flag = 0
        for i in range(0, len(title_list)):
            for k in final_list:
                if i+1 == k:
                    for j in range(0, len(title_list)):
                        if adjacency_matrix[i][j] == 1 and j+1 not in final_list:
                            final_list.append(j+1)
                            flag = 1

    print(final_list)
    #print(adjacency_matrix[i][j])
    print(results_list)
    for k in final_list:
        g.node('%02d' % k, shape='circle', color='red')

    for i in results_list:
        g.node('%02d' % i, shape='circle', color='blue')

    min_df = 4

    g.attr(label=r'\nGraph with search results for min_df = %d, number_topics =d, number_words =d and '
                 r'enrichment =d.\n Blue nodes were found in the search step in digital bases, red nodes were found '
                 r'through snowballing and black nodes were not found.' % min_df)
    g.attr(fontsize='12')

    r = graphviz.Source(g, filename="graph",
                        directory='/home/fuchs/Documentos/MESTRADO/Masters/Code/Exits/Snowballing/', format="ps")
    # r.render()
    r.view()

def main():
    """Main function."""
    results_list = [3]

    graph_snowballing(results_list)


if __name__ == "__main__":
    main()
