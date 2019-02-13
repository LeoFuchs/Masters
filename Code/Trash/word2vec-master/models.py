import torch
import torch.nn.functional as F
import numpy as np

TABLE_SIZE = 1e8

#Função para criar a tabela de amostras negativas do vocabulário
def create_sample_table(word_count):
    """ Cria uma tabela de amostras negativas para o vocabulário,
        palavras com maior frequência terão ocorrências mais altas na tabela.
    """

    table = []

    frequency = np.power(np.array(word_count), 0.75)
    sum_frequency = sum(frequency)
    ratio = frequency / sum_frequency
    count = np.round(ratio * TABLE_SIZE)

    for word_idx, c in enumerate(count):
        table += [word_idx] * int(c)

    return np.array(table)


#Função para executar o modelo Skip-gram
class SkipGramModel(torch.nn.Module):
    """ A palavra central é a entrada e as palavras de contexto são os alvos.
        O objetivo é maximizar a pontuação do mapa da entrada para o alvo.
    """

    def __init__(self, vocabulary_size, embedding_dim, neg_num = 0, word_count = []):
        super(SkipGramModel, self).__init__()
        self.neg_num = neg_num
        self.embeddings = torch.nn.Embedding(vocabulary_size, embedding_dim)
        initrange = 0.5 / embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        if self.neg_num > 0:
            self.table = create_sample_table(word_count)

    def forward(self, centers, contexts):
        batch_size = len(centers)

        u_embeds = self.embeddings(centers).view(batch_size, 1, -1)
        v_embeds = self.embeddings(contexts).view(batch_size, 1, -1)

        score  = torch.bmm(u_embeds, v_embeds.transpose(1,2)).squeeze()
        loss = F.logsigmoid(score).squeeze()

        if self.neg_num > 0:
            neg_contexts = torch.LongTensor(np.random.choice(self.table, size = (batch_size, self.neg_num)))
            neg_v_embeds = self.embeddings(neg_contexts)
            neg_score = torch.bmm(u_embeds, neg_v_embeds.transpose(1,2)).squeeze()
            neg_score = torch.sum(neg_score, dim = 1)
            neg_score = F.logsigmoid(-1 * neg_score).squeeze()
            loss += neg_score

        return -1 * loss.sum()

    def get_embeddings(self):
        return self.embeddings.weight.data

#Função para executar o modelo CBOW
class CBOWModel(torch.nn.Module):
    """ Palavras de contexto como entrada, retornando possibilidades de predição
        de distribuição da palavra central (alvo).
    """

    def __init__(self, vocabulary_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = torch.nn.Embedding(vocabulary_size, embedding_dim)
        initrange = 0.5 / embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.linear1 = torch.nn.Linear(embedding_dim, vocabulary_size)

    def forward(self, contexts):
        # Entrada
        embeds = self.embeddings(contexts)
        # Projeção
        add_embeds = torch.sum(embeds, dim = 1)
        # Saída
        out = self.linear1(add_embeds)
        log_probs = F.log_softmax(out, dim = 1)
        return log_probs

    def get_embeddings(self):
        return self.embeddings.weight.data
