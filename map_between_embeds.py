from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random

# I use small models because I'm GPU poor
model1 = "gpt2"
model2 = "EleutherAI/pythia-160m"

# Load the models
model1_tokenizer = AutoTokenizer.from_pretrained(model1)
model2_tokenizer = AutoTokenizer.from_pretrained(model2)

model1_vocab_dict = model1_tokenizer.get_vocab()
model2_vocab_dict = model2_tokenizer.get_vocab()
model1_vocab = set(model1_vocab_dict.keys())
model2_vocab = set(model2_vocab_dict.keys())
intersection = list(model1_vocab.intersection(model2_vocab))

# now we need to get the embedding matrix for each model
model1_model = AutoModelForCausalLM.from_pretrained(model1)
model2_model = AutoModelForCausalLM.from_pretrained(model2)

model1_embedding_matrix = model1_model.get_input_embeddings().weight.data
model2_embedding_matrix = model2_model.get_input_embeddings().weight.data

A = model1_embedding_matrix[model1_tokenizer.convert_tokens_to_ids(intersection)]
B = model2_embedding_matrix[model2_tokenizer.convert_tokens_to_ids(intersection)]

print(A.shape, B.shape)

# we need to find the average vector in A
A_avg = torch.mean(A, axis=0)
B_avg = torch.mean(B, axis=0)

# now we subtract the average vector from each vector in A and B
A = A - A_avg
B = B - B_avg

# split A and B into 90, 10
A_train = A[: int(0.9 * len(A))]
A_test = A[int(0.9 * len(A)) :]
B_train = B[: int(0.9 * len(B))]
B_test = B[int(0.9 * len(B)) :]

# then we need to create a matrix W
# then minimizing ||AW - B||^2

W = torch.linalg.lstsq(A_train, B_train).solution


test_mse = torch.mean((A_test @ W - B_test) ** 2)
train_mse = torch.mean((A_train @ W - B_train) ** 2)
# print(test_mse, train_mse)

# I now want to check how close this is for a random word, say " straw"

# pick a random word from intersections

sum = 0
num = 0

for i in range(100):
    word = random.choice(intersection)

    word_id1 = model1_tokenizer.convert_tokens_to_ids(word)
    word_id2 = model2_tokenizer.convert_tokens_to_ids(word)

    straw_embedding1 = model1_embedding_matrix[word_id1]
    straw_embedding2 = model2_embedding_matrix[word_id2]

    straw_embedding1 = straw_embedding1 - A_avg
    straw_embedding2 = straw_embedding2 - B_avg

    straw_embedding1 = straw_embedding1 @ W

    # get the dot product
    dot_product = torch.dot(
        straw_embedding1 / torch.linalg.norm(straw_embedding1),
        straw_embedding2 / torch.linalg.norm(straw_embedding2),
    )

    num += 1
    sum += dot_product

print(sum / num)
