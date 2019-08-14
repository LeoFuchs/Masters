# Font: https://huggingface.co/pytorch-transformers/quickstart.html
import torch
import numpy as np
from pytorch_transformers import BertTokenizer, BertForMaskedLM


word = 'organizations'

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input
text = "Over the last decade many organizations are increasingly concerned with the improvement of their " \
       "hardware/software development processes. The Capability Maturity Model and ISO9001 are well-known approaches " \
       "that are applied in these initiatives. However, one of the major bottlenecks to the success of process " \
       "improvement is the lack of business goal orientation. Additionally, business-oriented improvement approaches " \
       "often show a lack of process orientation. This paper reports on a process improvement initiative at Thales " \
       "Naval Netherlands that attempts to combine the best of both worlds, i.e. process improvement and business " \
       "goal orientation. Main factors in this approach are goal decomposition and the implementation of " \
       "goal-oriented measurement on three organizational levels, i.e. the business, the process and the team level. "

sentence = [sentence + '.' for sentence in text.split('.') if word in sentence][0]

sentence = "[CLS] " + sentence + " [SEP]"
print("Sentence: " + str(sentence))

tokenized_text = tokenizer.tokenize(sentence)
# print("Tokenized Text: " + str(tokenized_text))

len_segments_ids = len(tokenized_text)

# Defining the masked index equal the word of input

masked_index = 0

for count, token in enumerate(tokenized_text):
    if token == word:
        masked_index = count
        print ("Masked Index: " + str(masked_index))

        # Mask a token that we will try to predict back with `BertForMaskedLM`
        original_word = tokenized_text[masked_index]
        print("Original Word: " + str(original_word))

        tokenized_text[masked_index] = '[MASK]'
        # print("New Tokenized Text: " + str(tokenized_text))

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# print("Indexed Tokens: " + str(indexed_tokens))

# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0] * len_segments_ids
# print("Segments IDs: " + str(segments_ids))

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

# Predict the five first possibilities of the word removed
predicted_index = torch.topk(predictions[0, masked_index], 5)[1]
predicted_index = list(np.array(predicted_index))
# print("Predicted Index: " + str(predicted_index))

predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
predicted_token = [str(string) for string in predicted_token]
print("Predicted Word: " + str(predicted_token))

print(1)
