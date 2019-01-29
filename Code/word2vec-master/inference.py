import pickle

def save_embeddings(filename, embeddings, dictionary):
    """Armazenar em um arquivo os embeddings e a serialização do dicionário reverso."""

    data = {'emb':  embeddings, 'dict': dictionary}
    file = open(filename, 'wb')

    print("Salvando embeddings no arquivo:", filename)
    pickle.dump(data, file)

class Word2Vec(object):
    """Interface de inferencia de embeddings do Word2Vec
        Antes de inferir o resultado do embdedding de uma palavra, o dado precisa ser inicializado
        chamando o método from_file ou from_object.
    """

    def __init__(self):
        self.embeddings = None
        self.dictionary = None

    def from_file(self, filename):
        file = open(filename, 'rb')
        data = pickle.load(file)
        self.embeddings = data['emb']
        self.dictionary = data['dict']

    def from_object(self, embeddings, dictionary):
        self.embeddings = embeddings
        self.dictionary = dictionary

    def inference(self, word):
        assert self.embeddings is not None and self.dictionary is not None, 'Embeddings não inicializados, use from_file ou from_object para carregar o dado.'
        word_idx = self.dictionary.get(word)

        # # Palavra desconhecida retorna o word_idx do UNK
        if word_idx is None:
            word_idx = 0

        return self.embeddings[word_idx]