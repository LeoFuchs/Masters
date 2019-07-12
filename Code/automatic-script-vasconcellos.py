# encoding: utf-8
# run with python automatic-script-vasconcellos.py >> vasconcellos-out.txt
import gensim
import Levenshtein
import csv
import pandas as pd
import numpy as np
import graphviz
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from nltk.stem import LancasterStemmer
from graphviz import Graph
from pyscopus import Scopus


def bag_of_words(min_df, qgs_txt):
    """Gera a representação em bag-of-words do QGS.

    Args:

    Returns:

    """

    n_gram = (1, 3)
    max_document_frequency = 1.0
    min_document_frequency = min_df
    max_features = None

    # Carrega o dataset de treinamento (Está sempre na pasta Files-QGS)
    files = load_files(container_path=qgs_txt, encoding="iso-8859-1")

    # Extrai as palavras e vetoriza o dataset
    tf_vectorizer = CountVectorizer(max_df=max_document_frequency,
                                    min_df=min_document_frequency,
                                    ngram_range=n_gram,
                                    max_features=max_features,
                                    stop_words='english')

    tf = tf_vectorizer.fit_transform(files.data)

    # Salva os nomes das palavras em um dicionário
    dic = tf_vectorizer.get_feature_names()

    return dic, tf


def lda_algorithm(tf, lda_iterations, number_topics):
    """Executa o algoritmo LDA na representação de bag-of-words

    Args:

    Return:

    """

    alpha = None
    beta = None
    learning = 'batch'  # Batch ou Online

    # Executa o LDA e treina-o
    lda = LatentDirichletAllocation(n_components=number_topics,
                                    doc_topic_prior=alpha,
                                    topic_word_prior=beta,
                                    learning_method=learning,
                                    learning_decay=0.7,
                                    learning_offset=10.0,
                                    max_iter=lda_iterations,
                                    batch_size=128,
                                    evaluate_every=-1,
                                    total_samples=1000000.0,
                                    perp_tol=0.1,
                                    mean_change_tol=0.001,
                                    max_doc_update_iter=100,
                                    random_state=0)

    lda.fit(tf)

    return lda


def string_formulation(model, feature_names, number_words, number_topics, similar_words, levenshtein_distance, wiki,
                       pubyear):
    """Formula a string de busca baseada nos parâmetros de entrada

    Args:

    Return:

    """
    global final_similar_word
    message = "TITLE-ABS-KEY("
    if similar_words == 0:
        for topic_index, topic in enumerate(model.components_):
            message += "(\""
            message += "\" AND \"".join([feature_names[i] for i in topic.argsort()[:-number_words - 1:-1]])
            message += "\")"

            if topic_index < number_topics - 1:
                message += " OR "
            else:
                message += ""

        message += ")"

        if pubyear != 0:
            message += " AND PUBYEAR < "
            message += str(pubyear)

        return message

    else:
        lancaster = LancasterStemmer()

        word2vec_total_words = 30

        for topic_index, topic in enumerate(model.components_):
            counter = 0
            message += "("

            for i in topic.argsort()[:-number_words - 1:-1]:
                counter = counter + 1

                message += "(\""
                message += "\" - \"".join([feature_names[i]])

                if " " not in feature_names[i]:
                    try:
                        similar_word = wiki.most_similar(positive=feature_names[i], topn=word2vec_total_words)
                        similar_word = [j[0] for j in similar_word]
                        # print("Similar word:", similar_word)

                        stem_feature_names = lancaster.stem(feature_names[i])
                        # print("Stem feature names:", stem_feature_names)

                        stem_similar_word = []

                        final_stem_similar_word = []
                        final_similar_word = []

                        for j in similar_word:
                            stem_similar_word.append(lancaster.stem(j))
                        # print("Stem Similar Word:", stem_similar_word)

                        for number, word in enumerate(stem_similar_word):
                            if stem_feature_names != word and Levenshtein.distance(stem_feature_names,
                                                                                   word) > levenshtein_distance:
                                irrelevant = 0

                                for k in final_stem_similar_word:
                                    if Levenshtein.distance(k, word) < levenshtein_distance:
                                        irrelevant = 1

                                if irrelevant == 0:
                                    final_stem_similar_word.append(word)
                                    final_similar_word.append(similar_word[number])

                        # print("Final Stem Similar Word:", final_stem_similar_word)
                        # print("Final Similar Word:", final_similar_word)
                        # print("\n\n\n")

                        message += "\" OR \""
                        message += "\" OR \"".join(final_similar_word[m] for m in
                                                   range(0, similar_words))  # Where defined the number of similar words

                    except Exception as e:
                        print (e)

                message += "\")"

                if counter < len(topic.argsort()[:-number_words - 1:-1]):
                    message += " AND "
                else:
                    message += ""

            message += ")"

            if topic_index < number_topics - 1:
                message += " OR "
            else:
                message += ""

        message += ")"

        if pubyear != 0:
            message += " AND PUBYEAR < "
            message += str(pubyear)

        return message


def scopus_search(string):
    """Efetua a busca da string na Scopus

    Args:

    Returns:

    """
    results = 5000
    key = '7f59af901d2d86f78a1fd60c1bf9426a'
    scopus = Scopus(key)

    try:
        search_df = scopus.search(string, count=results, view='STANDARD', type_=1)
        # print("Number of results without improvement:", len(search_df))
    except Exception as e:
        print (e)
        return -1

    pd.options.display.max_rows = 99999
    pd.options.display.max_colwidth = 250

    search_df[['title']].to_csv("/home/fuchs/Documentos/MESTRADO/Masters/Code/Exits/Result.csv", index_label=False,
                                encoding='utf-8', index=False, header=True, sep='\t')

    return int(len(search_df))


def open_necessary_files():
    """Abre os arquivos que serão utilizados

    Args:

    Returns:

    """
    gs = pd.read_csv('/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-vasconcellos/GS.csv', sep='\t')

    qgs = pd.read_csv('/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-vasconcellos/QGS.csv', sep='\t')

    result_name_list = pd.read_csv('/home/fuchs/Documentos/MESTRADO/Masters/Code/Exits/Result.csv', sep='\t')
    result_name_list = result_name_list.fillna(' ')

    manual_comparation = open('/home/fuchs/Documentos/MESTRADO/Masters/Code/Exits/ManualExit.csv', 'w')

    return qgs, gs, result_name_list, manual_comparation


def similarity_score_qgs(qgs, result_name_list, manual_comparation):
    """Faz a comparação automática entre o QGS e os resultados, obtendo a contagem
    de artigos do QGS presentes no resultado

    Args:

    Returns:

    """
    len_qgs = sum(
        1 for line in open('/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-vasconcellos/QGS.csv')) - 1
    len_result = sum(1 for line in open('/home/fuchs/Documentos/MESTRADO/Masters/Code/Exits/Result.csv')) - 1

    list_qgs = []
    list_result = []

    counter_improvement = 0

    for i in range(0, len_qgs):
        list_qgs.append(qgs.iloc[i, 0].lower())

    # print("Lista qgs:", list_qgs)
    # print("Tamanho Lista qgs:", len(list_qgs))

    for i in range(0, len_result):
        list_result.append(result_name_list.iloc[i, 0].lower())

    if len_result == 0:
        return counter_improvement

    # print("Lista Resultado:", list_result)
    # print("Tamanho Lista Resultado:", len(list_result))

    train_set = [list_qgs, list_result]
    train_set = [val for sublist in train_set for val in sublist]

    # print("Lista train_set:", train_set)
    # print("Elementos train_set", len(train_set))

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)

    mat_similaridade = cosine_similarity(tfidf_matrix_train[0:len_qgs],
                                         tfidf_matrix_train[len_qgs:len_qgs + len_result])
    lin, col = mat_similaridade.shape

    for i in range(0, lin):

        line = mat_similaridade[i]
        current_nearest = np.argsort(line)[-2:]  # Pega os x - 1 maiores elementos

        line_exit = 'QGS' + str(i + 1) + ':\t\t\t' + list_qgs[i] + '\t' + '\n'

        for j in range(1, len(current_nearest)):
            book = current_nearest[-j]
            line_exit = line_exit + '\t\t\t\t' + list_result[book].strip() + '\t' '\n'

            if Levenshtein.distance(list_qgs[i], list_result[book]) < 10:
                counter_improvement = counter_improvement + 1

        line_exit = line_exit + "\n"

        manual_comparation.write(line_exit)
        manual_comparation.flush()

    # print("Number of QGS articles founded (with improvement):", counter_improvement)

    return counter_improvement


def similarity_score_gs(gs, result_name_list, manual_comparation):
    """Faz a comparação automática entre o GS e os resultados,
    obtendo a contagem de artigos do GS presentes no resultado

    Args:

    Returns:

    """

    len_gs = sum(
        1 for line in open('/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-vasconcellos/GS.csv')) - 1
    len_result = sum(1 for line in open('/home/fuchs/Documentos/MESTRADO/Masters/Code/Exits/Result.csv')) - 1

    list_gs = []
    list_result = []

    list_graph = []

    counter_improvement = 0

    for i in range(0, len_gs):
        list_gs.append(gs.iloc[i, 0].lower())

    # print("Lista GS:", list_gs)
    # print("Tamanho Lista GS:", len(list_gs))

    for i in range(0, len_result):
        list_result.append(result_name_list.iloc[i, 0].lower())

    if len_result == 0:
        return counter_improvement

    # print("Lista Resultado:", list_result)
    # print("Tamanho Lista Resultado:", len(list_result))

    train_set = [list_gs, list_result]
    train_set = [val for sublist in train_set for val in sublist]

    # print("Lista train_set:", train_set)
    # print("Elementos train_set", len(train_set))

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)

    mat_similaridade = cosine_similarity(tfidf_matrix_train[0:len_gs], tfidf_matrix_train[len_gs:len_gs + len_result])
    lin, col = mat_similaridade.shape

    for i in range(0, lin):

        line = mat_similaridade[i]
        current_nearest = np.argsort(line)[-2:]  # Pega os x - 1 maiores elementos

        line_exit = 'GS' + str(i + 1) + ':\t\t\t' + list_gs[i] + '\t' + '\n'

        for j in range(1, len(current_nearest)):
            book = current_nearest[-j]
            line_exit = line_exit + '\t\t\t\t' + list_result[book].strip() + '\t' '\n'

            if Levenshtein.distance(list_gs[i], list_result[book]) < 10:
                counter_improvement = counter_improvement + 1
                list_graph.append(i + 1)

        line_exit = line_exit + "\n"

        manual_comparation.write(line_exit)
        manual_comparation.flush()

    # print("Number of GS articles founded (with improvement):", counter_improvement)

    return counter_improvement, list_graph


def graph_vasconcellos(list, min_df, number_topics, number_words, enrichment):
    """# Gera o grafo que apresenta quais artigos do GS foram encontrados ao se analisar os resultados

    Args:

    Returns:

    """

    g = Graph('Vasconcellos Graph', strict=True)

    for i in list:
        g.attr('node', shape='doublecircle')
        g.node('%02d' % i)

    # List append apenas nos filhos dos filhos nas arvores

    for i in list:
        if i == 1:
            pass
        if i == 2:
            g.edge('02', '03')
            g.edge('02', '16')
            g.edge('02', '17')
            g.edge('02', '21')
            if 18 not in list:
                list.append(18)
            if 20 not in list:
                list.append(20)
        if i == 3:
            g.edge('03', '02')
            if 16 not in list:
                list.append(16)
            if 17 not in list:
                list.append(17)
            if 21 not in list:
                list.append(21)

        if i == 4:
            g.edge('04', '15')
            g.edge('04', '18')
            g.edge('04', '19')
            if 5 not in list:
                list.append(5)
            if 14 not in list:
                list.append(14)
            if 15 not in list:
                list.append(15)
            if 17 not in list:
                list.append(17)
            if 18 not in list:
                list.append(18)
        if i == 5:
            g.edge('05', '18')
            if 4 not in list:
                list.append(4)
            if 15 not in list:
                list.append(15)
            if 17 not in list:
                list.append(17)
        if i == 6:
            pass
        if i == 7:
            pass
        if i == 8:
            g.edge('08', '09')
            g.edge('08', '14')
            if 9 not in list:
                list.append(9)
            if 12 not in list:
                list.append(12)
            if 13 not in list:
                list.append(13)
            if 14 not in list:
                list.append(14)
            if 19 not in list:
                list.append(19)
            if 29 not in list:
                list.append(29)
        if i == 9:
            g.edge('09', '08')
            g.edge('09', '13')
            g.edge('09', '14')
            if 8 not in list:
                list.append(8)
            if 12 not in list:
                list.append(12)
            if 14 not in list:
                list.append(14)
            if 19 not in list:
                list.append(19)
            if 29 not in list:
                list.append(29)
        if i == 10:
            g.edge('10', '11')
            g.edge('10', '30')
            if 24 not in list:
                list.append(24)
            if 25 not in list:
                list.append(25)
        if i == 11:
            g.edge('11', '10')
            if 30 not in list:
                list.append(30)
        if i == 12:
            g.edge('12', '13')
            g.edge('12', '14')
            if 8 not in list:
                list.append(8)
            if 9 not in list:
                list.append(9)
            if 19 not in list:
                list.append(19)
            if 29 not in list:
                list.append(29)
        if i == 13:
            g.edge('13', '09')
            g.edge('13', '12')
            if 8 not in list:
                list.append(8)
            if 14 not in list:
                list.append(14)
        if i == 14:
            g.edge('14', '08')
            g.edge('14', '09')
            g.edge('14', '12')
            g.edge('14', '19')
            g.edge('14', '29')
            if 4 not in list:
                list.append(4)
            if 8 not in list:
                list.append(8)
            if 9 not in list:
                list.append(9)
            if 13 not in list:
                list.append(13)
        if i == 15:
            g.edge('15', '04')
            g.edge('15', '18')
            if 4 not in list:
                list.append(4)
            if 5 not in list:
                list.append(5)
            if 18 not in list:
                list.append(18)
            if 19 not in list:
                list.append(19)
            if 17 not in list:
                list.append(17)
        if i == 16:
            g.edge('16', '02')
            if 3 not in list:
                list.append(3)
            if 17 not in list:
                list.append(17)
            if 21 not in list:
                list.append(21)
        if i == 17:
            g.edge('17', '02')
            g.edge('17', '18')
            if 3 not in list:
                list.append(3)
            if 4 not in list:
                list.append(4)
            if 5 not in list:
                list.append(5)
            if 15 not in list:
                list.append(15)
            if 16 not in list:
                list.append(16)
            if 21 not in list:
                list.append(21)
        if i == 18:
            g.edge('18', '04')
            g.edge('18', '05')
            g.edge('18', '15')
            g.edge('18', '17')
            if 2 not in list:
                list.append(2)
            if 4 not in list:
                list.append(4)
            if 15 not in list:
                list.append(15)
            if 19 not in list:
                list.append(19)
        if i == 19:
            g.edge('19', '04')
            g.edge('19', '14')
            if 8 not in list:
                list.append(8)
            if 9 not in list:
                list.append(9)
            if 12 not in list:
                list.append(12)
            if 15 not in list:
                list.append(15)
            if 18 not in list:
                list.append(18)
            if 29 not in list:
                list.append(29)
        if i == 20:
            g.edge('20', '21')
            if 2 not in list:
                list.append(2)
        if i == 21:
            g.edge('21', '02')
            g.edge('21', '20')
            if 3 not in list:
                list.append(3)
            if 16 not in list:
                list.append(16)
            if 17 not in list:
                list.append(17)
        if i == 22:
            g.edge('22', '28')
        if i == 23:
            pass
        if i == 24:
            g.edge('24', '30')
            if 10 not in list:
                list.append(10)
            if 25 not in list:
                list.append(25)
        if i == 25:
            g.edge('25', '30')
            if 10 not in list:
                list.append(10)
            if 24 not in list:
                list.append(24)
        if i == 26:
            pass
        if i == 27:
            pass
        if i == 28:
            g.edge('28', '22')
        if i == 29:
            g.edge('29', '14')
            if 8 not in list:
                list.append(8)
            if 9 not in list:
                list.append(9)
            if 12 not in list:
                list.append(12)
            if 19 not in list:
                list.append(19)
        if i == 30:
            g.edge('30', '10')
            g.edge('30', '24')
            g.edge('30', '25')
            if 11 not in list:
                list.append(11)

    for i in range(1, 31):
        g.attr('node', shape='circle')
        g.node('%02d' % i)

    g.edge('02', '03')
    g.edge('08', '09')
    g.edge('10', '30')
    g.edge('11', '10')
    g.edge('14', '09')
    g.edge('14', '12')
    g.edge('15', '04')
    g.edge('15', '18')
    g.edge('16', '02')
    g.edge('17', '02')
    g.edge('17', '18')
    g.edge('18', '04')
    g.edge('18', '05')
    g.edge('19', '04')
    g.edge('19', '14')
    g.edge('21', '02')
    g.edge('21', '20')
    g.edge('24', '30')
    g.edge('25', '30')
    g.edge('28', '22')
    g.edge('29', '14')

    for i in range(1, 31):
        g.attr('node', shape='circle')
        g.node('%02d' % i)

    g.edge('02', '03')
    g.edge('08', '09')
    g.edge('10', '30')
    g.edge('11', '10')
    g.edge('14', '09')
    g.edge('14', '12')
    g.edge('15', '04')
    g.edge('15', '18')
    g.edge('16', '02')
    g.edge('17', '02')
    g.edge('17', '18')
    g.edge('18', '04')
    g.edge('18', '05')
    g.edge('19', '04')
    g.edge('19', '14')
    g.edge('21', '02')
    g.edge('21', '20')
    g.edge('24', '30')
    g.edge('25', '30')
    g.edge('28', '22')
    g.edge('29', '14')

    r = graphviz.Source(g, filename="graph-with-%0.1f-%d-%d-%d" % (min_df, number_topics, number_words, enrichment),
                        directory='/home/fuchs/Documentos/MESTRADO/Masters/Code/Exits/Graphs/', format="ps")
    r.render()


def graph_snowballing(results_list, min_df, number_topics, number_words, enrichment):
    """Gera o grafo que apresenta o snowballing do GS e quais artigos
    do GS foram encontrados ao se analisar os resultados

    Args:

    Returns:

    """

    with open('/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-vasconcellos/GS.csv', mode='r') as gs:

        # Pulando a linha do GS.csv escrita 'title'
        next(gs)

        # Criando uma lista onde cada elemento é o nome de um artigo do GS, sem espaços, minusculos e sem -
        title_list = [line.strip().lower().replace(' ', '').replace('-', '') for line in gs]

        # Criando uma lista auxiliar de tamanho n no formato [1, 2, 3, 4, 5, ... , n]
        node_list = range(1, len(title_list) + 1)

    gs.close()

    # Inicializando o Grafo com os seus respectivos nós
    g = Graph('Snowballing Graph', strict=True)
    for i in node_list:
        g.node('%02d' % i, shape='circle')

    # Analisando a citação de cada um dos artigos
    for i in range(1, len(title_list) + 1):
        article_name = '/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-vasconcellos/GS-pdf/%d.cermzones' % i
        with open('%s' % article_name, mode='r') as file_zone:

            # Manipulando o arquivo de entrada (RETIRAR ESSE I != 16 DEPOIS)
            if i != 16:
                # Tornando todas as letras minusculas
                reader = file_zone.read().lower()

                # Removendo as quebras de linha
                reader = reader.strip().replace('\n', ' ').replace('\r', '')

                # Removendo os espaços e caracteres especiais
                reader = reader.replace(' ', '').replace('-', '')

                # Filtrando apenas a parte das referências
                sep = "<zonelabel=\"gen_references\">"
                reader = reader.split(sep, 1)[1]
                # print(reader)

                for j in range(1, len(title_list) + 1):
                    if i != j:
                        if title_list[j - 1] in reader:
                            # print("O artigo GS-%02.d cita o artigo %02.d.\n" % (i, j))
                            g.edge('%02d' % i, '%02d' % j)
                            g.edge('%02d' % j, '%02d' % i)
        file_zone.close()

    for i in results_list:
        g.node('%02d' % i, shape='circle', color='red')

    r = graphviz.Source(g, filename="graph-with-%0.1f-%d-%d-%d" % (min_df, number_topics, number_words, enrichment),
                        directory='/home/fuchs/Documentos/MESTRADO/Masters/Code/Exits/Snowballing/', format="ps")
    r.render()
    # r.view()


def main():
    levenshtein_distance = 4
    lda_iterations = 5000

    qgs_txt = '/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-vasconcellos/QGS-txt/metadata'

    pubyear = 2015  # Pubyear with 0 = disable

    min_df_list = [0.4]
    number_topics_list = [3]
    number_words_list = [7]
    enrichment_list = [0, 1]

    print("Loading wiki...\n")
    wiki = gensim.models.KeyedVectors.load_word2vec_format(
        '/home/fuchs/Documentos/MESTRADO/Datasets/wiki-news-300d-1M.vec')

    # Rodando o CERMINE
    print("Loading CERMINE...\n")
    cermine = "java -cp cermine-impl-1.13-jar-with-dependencies.jar pl.edu.icm.cermine.ContentExtractor -path " \
              "/home/fuchs/Documentos/MESTRADO/Masters/Files-QGS/revisao-vasconcellos/GS-pdf/ -outputs zones "
    os.system(cermine)

    with open('vasconcellos-output.csv', mode='w') as file_output:

        file_writer = csv.writer(file_output, delimiter=',')

        file_writer.writerow(['min_df', 'Topics', 'Words', 'Similar Words', 'No. Results', 'No. QGS', 'No. GS'])

        for min_df in min_df_list:
            for number_topics in number_topics_list:
                for number_words in number_words_list:

                    print("Test with " + str(number_topics) + " topics and " + str(number_words) + " words in " + str(
                        min_df) + " min_df:")
                    print("\n")

                    dic, tf = bag_of_words(min_df, qgs_txt)
                    lda = lda_algorithm(tf, lda_iterations, number_topics)

                    for enrichment in enrichment_list:
                        string = string_formulation(lda, dic, number_words, number_topics, enrichment,
                                                    levenshtein_distance, wiki, pubyear)

                        scopus_number_results = scopus_search(string)

                        qgs, gs, result_name_list, manual_comparation = open_necessary_files()
                        counter_one = similarity_score_qgs(qgs, result_name_list, manual_comparation)
                        counter_two, list_graph = similarity_score_gs(gs, result_name_list, manual_comparation)

                        file_writer.writerow(
                            [min_df, number_topics, number_words, enrichment, scopus_number_results, counter_one,
                             counter_two])

                        print("String with " + str(enrichment) + " similar words: " + str(string))
                        print("Generating " + str(scopus_number_results) + " results with " + str(
                            counter_one) + " of the QGS articles and " + str(counter_two) + " of the GS articles.")
                        print("\n")

                        # graph_vasconcellos(list_graph, min_df, number_topics, number_words, enrichment)
                        graph_snowballing(list_graph, min_df, number_topics, number_words, enrichment)
    file_output.close()


if __name__ == "__main__":
    main()
