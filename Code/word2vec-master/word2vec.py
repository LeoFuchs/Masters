import argparse
import zipfile
import re
import collections
import numpy as np
from gensim.models import KeyedVectors
from six.moves import xrange
import random
import torch
import gensim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.autograd import Variable
from models import SkipGramModel
from models import CBOWModel
from inference import Word2Vec
from inference import save_embeddings


from gensim.test.utils import datapath

model_list = ['CBOW', 'skipgram']

cmd_parser = argparse.ArgumentParser(description = None)

# Argumentos

#Dados de Entrada
cmd_parser.add_argument('-d', '--data', default = 'data/2.txt', help = 'Data file for word2vec training.')

#Dados de Saida
cmd_parser.add_argument('-o', '--output', default = 'embeddings.bin', help = 'Output embeddings filename.')

#Imagem de Saida
cmd_parser.add_argument('-p', '--plot', default = 'tsne.png', help = 'Plotting output filename.')

#Configurações da Imagem de Saida
cmd_parser.add_argument('-pn', '--plot_num', default = 3, type = int, help = 'Plotting data number.')

#Tamanho do Vocabulário
cmd_parser.add_argument('-s', '--size', default = 50000, type = int, help = 'Vocabulary size.')


# Argumentos do Modelo de Treinamento
cmd_parser.add_argument('-m', '--mode', default = 'CBOW', choices = model_list, help = 'Training model.')
cmd_parser.add_argument('-bs', '--batch_size', default = 128, type = int, help = 'Training batch size.')
cmd_parser.add_argument('-ns', '--num_skips', default = 2, type = int, help = 'How many times to reuse an input to generate a label.')
cmd_parser.add_argument('-sw', '--skip_window', default = 2, type = int, help = 'How many words to consider left and right.')
cmd_parser.add_argument('-ed', '--embedding_dim', default = 128, type = int, help = 'Dimension of the embedding vector.')
cmd_parser.add_argument('-lr', '--learning_rate', default = 0.025, type = float, help = 'Learning rate')
cmd_parser.add_argument('-i', '--num_steps', default = 50, type = int, help = 'Number of steps to run.')
cmd_parser.add_argument('-ne', '--negative_example', default = 5, type = int, help = 'Number of negative examples.')


#Função que faz a leitura dos dados
def read_data(filename):
    """Extrai o primeiro arquivo contido em um arquivo .zip com uma lista de palavras, ou então extrai diretamente do arquivo .txt."""
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename) as f:
            text = f.read(f.namelist()[0]).decode('ascii')
    else:
        with open(filename, "r") as f:
            text = f.read()
    return [word.lower() for word in re.compile('\w+').findall(text)]


#Função que cria o dataset
def build_dataset(words, n_words):
    """Processar as entradas brutas do dataset.
        Retorna:
            data:               Lista de códigos (inteiros de 0 a vocabulary_size - 1).
                                Este é o texto original, mas as palavras são substituídas por seus códigos
            count:              Lista de palavras (strings) para contagem de ocorrências
            dictionary:         Mapa de palavras (strings) para seus códigos (inteiros)
            reverse_dictionary: Mapa de códigos (inteiros) para palavras (strings)
    """

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()

    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0

    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)

    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reversed_dictionary


#Função que gera um lote de dados de treinamento
def generate_batch(data, data_index, batch_size, num_skips, skip_window):
    """Gera um lote de dados de treinamento
        Retorna:
            centers:        Lista de índices de palavras centrais para esse lote.
            contexts:       Lista de índices de contextos para esse lote.
            data_index:     Indice dos dados atuais para o próximo lote.
    """

    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    centers = np.ndarray(shape = (batch_size), dtype = np.int32)
    contexts = np.ndarray(shape = (batch_size, 1), dtype = np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    if data_index + span > len(data):
        data_index = 0

    buffer.extend(data[data_index:data_index + span])
    data_index += span

    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            centers[i * num_skips + j] = buffer[skip_window]
            contexts[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            for word in data[:span]:
                buffer.append(word)
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    # Voltar um pouco para evitar pular palavras no final de um lote
    data_index = (data_index + len(data) - span) % len(data)

    return torch.LongTensor(centers), torch.LongTensor(contexts), data_index

#Função que faz o treinamento
def train(data, word_count, mode, vocabulary_size, embedding_dim, batch_size, num_skips, skip_window, num_steps, learning_rate, neg_num):
    """Processo de Treinamento e Backpropagation, retorna o embedding final como result"""

    if mode == 'CBOW':
        model = CBOWModel(vocabulary_size, embedding_dim)
    elif mode == 'skipgram':
        model = SkipGramModel(vocabulary_size, embedding_dim, neg_num, word_count)
    else:
        raise ValueError("Modelo \"%s\" não é suportado" % model)

    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    loss_function = torch.nn.NLLLoss()
    data_index = 0
    loss_val = 0

    for i in xrange(num_steps):
        # Preparar dados de feed e forward pass
        centers, contexts, data_index = generate_batch(data, data_index, batch_size, num_skips, skip_window)
        if mode == 'CBOW':
            y_pred = model(contexts)
            loss = loss_function(y_pred, centers)
        elif mode == 'skipgram':
            loss = model(centers, contexts)
        else:
            raise ValueError("Modelo \"%s\" não é suportado" % model)

        # Gradientes Zero, execute um backward pass, e atualize os pesos.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Imprimir o valor da perda para certos passos
        loss_val += loss.item()
        if i > 0 and i % (num_steps/100) == 0:
            print('Perda média no passo', i, ':', loss_val/(num_steps/100))
            loss_val = 0

    return model.get_embeddings()


#Função que Plota a Imagem
def tsne_plot(embeddings, num, reverse_dictionary, filename):
    """Plotar o resultado tSNE de embeddings para um subconjunto de palavras"""

    tsne = TSNE(perplexity = 30, n_components = 2, init='pca', n_iter = 5000, method='exact')
    low_dim_embs = tsne.fit_transform(final_embeddings[:num, :])
    low_dim_labels = [reverse_dictionary[i] for i in xrange(num)]

    assert low_dim_embs.shape[0] >= len(low_dim_labels), 'More labels than embeddings'

    plt.figure(figsize=(18, 18))  # Em polegadas

    for i, label in enumerate(low_dim_labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy = (x, y), xytext = (5, 2), textcoords = 'offset points', ha = 'right', va = 'bottom')
    print("Salvando imagem no arquivo:", filename)
    plt.savefig(filename)

#Função Main
if __name__ == '__main__':

    args = cmd_parser.parse_args()

    # Pré-processamento dos dados
    vocabulary = read_data(args.data)
    print('Tamanho dos Dados:', len(vocabulary))
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, args.size)

    vocabulary_size = min(args.size, len(count))
    print('Tamanho do Vocabulário:', vocabulary_size)

    word_count = [c[1] for c in count]

    # Treinamento do Modelo
    final_embeddings = train(data = data,
                             word_count = word_count,
                             mode = args.mode,
                             vocabulary_size = vocabulary_size,
                             embedding_dim = args.embedding_dim,
                             batch_size = args.batch_size,
                             num_skips = args.num_skips,
                             skip_window = args.skip_window,
                             num_steps = args.num_steps,
                             learning_rate = args.learning_rate,
                             neg_num = args.negative_example)

    norm = torch.sqrt(torch.cumsum(torch.mul(final_embeddings, final_embeddings), 1))
    nomalized_embeddings = (final_embeddings/norm).numpy()

    # Salvando resultados e plotando imagem
    save_embeddings(args.output, final_embeddings, dictionary)
    tsne_plot(embeddings = nomalized_embeddings, num = min(vocabulary_size, args.plot_num), reverse_dictionary = reverse_dictionary, filename = args.plot)

    w2v = Word2Vec()
    w2v.from_file('embeddings.bin')

    word = 'the'
    print('Embeddings da palavra:', word)
    print(w2v.inference('the'))

    word = 'of'
    print('Embeddings da palavra:', word)
    print(w2v.inference('of'))

    model = KeyedVectors.load_word2vec_format(datapath(('/usr/local/lib/python3.6/dist-packages/gensim/test/test_data/euclidean_vectors.bin')), binary = True)  # C binary format
    stringA = 'of'
    stringB = 'the'

    print(model.most_similar(positive=[stringA, stringB], negative=[stringA, stringB], topn=10))

    model = KeyedVectors.load_word2vec_format(datapath(('/home/fuchs/Documentos/MESTRADO/Masters/Code/word2vec-master/embeddings.bin')), encoding='utf8', unicode_errors='ignore', binary=True)  # C binary format
    stringA = 'of'
    stringB = 'the'

    print(model.most_similar(positive=[stringA, stringB], negative=[stringA, stringB], topn=10))