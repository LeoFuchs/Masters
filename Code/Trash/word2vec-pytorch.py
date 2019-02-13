# see http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
print("============================================================")

CONTEXT_SIZE = 2  # Janela de contexto com 2 palavras para a esquerda e 2 palavras para a direita

# Formaremos um vetor onde cada posição será uma palavra
train = """Procuring education on consulted assurance in do. Is sympathize he expression mr no travelling. 
Preference he he at travelling in resolution. So striking at of to welcomed resolved. 
Northward by described up household therefore attention. Excellence decisively nay man yet impression for contrasted remarkably.
There spoke happy for you are out. Fertile how old address did showing because sitting replied six. 
Had arose guest visit going off child she new.""".split()

test = """So the striking at of to welcomed resolved.""".split()

# Vamos formular o vocabulario do texto unindo as palavras repetidas do texto.
vocab = set(train)
print('Vocabulary:', vocab)
print("============================================================")

vocab_size = len(vocab)
print('Vocabulary Size:', vocab_size)
print("============================================================")

w2i = {w: i for i, w in enumerate(vocab)} # word to index (palavra : indice)
i2w = {i: w for i, w in enumerate(vocab)} # index to word (indice : palavra)

#Criando o dataset para o CBOW
def create_cbow_dataset(text):
    data = []
    for i in range(2, len(text) - 2):
        context = [text[i - 2], text[i - 1], text[i + 1], text[i + 2]] #Contexto são as duas palavras antes e as duas palavras depois
        target = text[i] # Alvo é a palavra atual no laço
        data.append((context, target))
    return data

#Criando o dataset para o Skip-gram
def create_skipgram_dataset(text):
    import random
    data = []
    for i in range(2, len(text) - 2):
        data.append((text[i], text[i-2], 1))
        data.append((text[i], text[i-1], 1))
        data.append((text[i], text[i+1], 1))
        data.append((text[i], text[i+2], 1))
        # Negative Sampling
        for _ in range(4):
            if random.random() < 0.5 or i >= len(text) - 3:
                rand_id = random.randint(0, i-1)
            else:
                rand_id = random.randint(i+3, len(text)-1)
            data.append((text[i], text[rand_id], 0))
    return data

cbow_train = create_cbow_dataset(train)
skipgram_train = create_skipgram_dataset(train)

print('Amostra CBOW Treino:')
for i in range(0, 5):
  print(cbow_train[i])
print("============================================================")

print('Amostra Skip-Gram Treino:')
for i in range(0, 8):
  print(skipgram_train[i])
print("============================================================")

cbow_test = create_cbow_dataset(test)
skipgram_test = create_skipgram_dataset(test)

print('Amostra CBOW Teste:')
for i in range(0, 3):
  print(cbow_test[i])
print("============================================================")

print('Amostra Skip-Gram Teste:')
for i in range(0, 8):
  print(skipgram_test[i])
print("============================================================")

class CBOW(nn.Module):
  def __init__(self, vocab_size, embd_size, context_size, hidden_size):
    super(CBOW, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embd_size)
    self.linear1 = nn.Linear(2 * context_size * embd_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, vocab_size)

  def forward(self, inputs):
    embedded = self.embeddings(inputs).view((1, -1))
    hid = F.relu(self.linear1(embedded))
    out = self.linear2(hid)
    log_probs = F.log_softmax(out, dim=1)
    return log_probs


class SkipGram(nn.Module):
  def __init__(self, vocab_size, embd_size):
    super(SkipGram, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embd_size)

  def forward(self, focus, context):
    embed_focus = self.embeddings(focus).view((1, -1))
    embed_ctx = self.embeddings(context).view((1, -1))
    score = torch.mm(embed_focus, torch.t(embed_ctx))
    log_probs = F.logsigmoid(score)

    return log_probs


embd_size = 100 #Tamanho dos Embeddings
learning_rate = 0.001 #Taxa de Aprendizado
n_epoch = 30 #Número de Épocas

#Treinando o CBOW
def train_cbow():
  hidden_size = 64
  losses = []
  loss_fn = nn.NLLLoss()
  model = CBOW(vocab_size, embd_size, CONTEXT_SIZE, hidden_size)
  optimizer = optim.SGD(model.parameters(), lr=learning_rate)

  for epoch in range(n_epoch):
    total_loss = .0
    for context, target in cbow_train:
      ctx_idxs = [w2i[w] for w in context]
      ctx_var = Variable(torch.LongTensor(ctx_idxs))

      model.zero_grad()
      log_probs = model(ctx_var)

      loss = loss_fn(log_probs, Variable(torch.LongTensor([w2i[target]])))

      loss.backward()
      optimizer.step()

      total_loss += loss.item()
    losses.append(total_loss)
  return model, losses


#Treinando Skip-gram
def train_skipgram():
  losses = []
  loss_fn = nn.MSELoss()
  model = SkipGram(vocab_size, embd_size)
  optimizer = optim.SGD(model.parameters(), lr=learning_rate)

  for epoch in range(n_epoch):
    total_loss = .0
    for in_w, out_w, target in skipgram_train:
      in_w_var = Variable(torch.LongTensor([w2i[in_w]]))
      out_w_var = Variable(torch.LongTensor([w2i[out_w]]))

      model.zero_grad()
      log_probs = model(in_w_var, out_w_var)
      loss = loss_fn(log_probs[0], Variable(torch.Tensor([target])))

      loss.backward()
      optimizer.step()

      total_loss += loss.item()
    losses.append(total_loss)
  return model, losses


cbow_model, cbow_losses = train_cbow()
sg_model, sg_losses = train_skipgram()


# Testes
# Deve-se usar outro conjunto de dados para teste!

# Teste CBOW
def test_cbow(test_data, model):
  print('Teste CBOW:\n')
  correct_ct = 0
  for ctx, target in test_data:
    ctx_idxs = [w2i[w] for w in ctx]
    ctx_var = Variable(torch.LongTensor(ctx_idxs))

    model.zero_grad()
    log_probs = model(ctx_var)
    _, predicted = torch.max(log_probs.data, 1)
    predicted_word = i2w[predicted.item()]
    print('Previsão:', predicted_word)
    print('Palavra:', target)
    if predicted_word == target:
      correct_ct += 1

  print('\n')
  print('Precisão: {:.1f}% ({:d}/{:d})'.format(correct_ct / len(test_data) * 100, correct_ct, len(test_data)))
  print("============================================================")

# Teste Skip-gram
def test_skipgram(test_data, model):
  print('Teste Skip-Gram:\n')
  correct_ct = 0
  for in_w, out_w, target in test_data:
    in_w_var = Variable(torch.LongTensor([w2i[in_w]]))
    out_w_var = Variable(torch.LongTensor([w2i[out_w]]))

    model.zero_grad()
    log_probs = model(in_w_var, out_w_var)
    _, predicted = torch.max(log_probs.data, 1)
    predicted = predicted.item()
    print('Previsão:', predicted)
    print('Palavra:', target)
    if predicted == target:
      correct_ct += 1

  print('\n')
  print('Precisão: {:.1f}% ({:d}/{:d})'.format(correct_ct / len(test_data) * 100, correct_ct, len(test_data)))
  print("============================================================")


test_cbow(cbow_test, cbow_model)
test_skipgram(skipgram_test, sg_model)


def print_k_nearest_neighbour(X, idx, k, idx_to_word):
  """
  :param X: Embedding Matrix |V x D|
  :param idx: The Knn of the ith
  :param k: k nearest neighbour
  :return:
  """

  dists = np.dot((X - X[idx]) ** 2, np.ones(X.shape[1]))
  idxs = np.argsort(dists)[:k]

  print('O(s) {} vizinhos mais próximos da palavra \'{}\' são: '.format(str(k), idx_to_word[idx]))
  for i in idxs:
    print('\'' + idx_to_word[i] + '\'', end=' ')

  print('\n')
  print("============================================================")
  return idxs

embed_matrix = cbow_model.embeddings.weight.detach().cpu().numpy()
print(embed_matrix)
print_k_nearest_neighbour(embed_matrix, w2i['going'], 5, i2w)
