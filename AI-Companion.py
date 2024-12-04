from gensim.downloader import load
import numpy as np
import math 

dims = 300

## 300 dimensions TODO: Download locally/or database
glove_model = load(f'glove-wiki-gigaword-{dims}')



## Temporary function until api
def encode_input():
    user_input = input("USER: ")
    sentence = user_input.split()
    embeddings = []

    for message in sentence:
        if message in glove_model:
            embeddings.append(glove_model[message])
        else: 
            print(f"Words could not be embedded") ##TODO Randomize here.
    return np.array(embeddings)

def decode_output(vector):
    closest_similarity = -1
    closest_word = None

    for word in glove_model.key_to_index:
        similarity = np.dot(glove_model[word], vector) / (np.linalg.norm(glove_model[word])
        * np.linalg.norm(vector))
        if similarity > closest_similarity:
            closest_similarity = similarity
            closest_word = word
    return closest_word

## UTIL functions

def softmax(matrix):
    exp = np.exp(matrix - np.max(matrix))
    return exp / np.sum(exp, axis=  -1, keepdims=True)

## static values for dims later?
def random_matrix(rows,cols):
    rand_matrix = np.random.rand(rows, cols)
    return rand_matrix


def attention():
    input_matrix = encode_input()
    
    cols = 300

    W_Q = random_matrix(dims,cols)
    W_K = random_matrix(dims,cols)
    W_V = random_matrix(dims,cols)

    query = np.dot(input_matrix, W_Q)
    key = np.dot(input_matrix, W_K)
    value = np.dot(input_matrix, W_V)

    raw_attention_score = np.dot(query,key.T)

    d_k = query.shape[1]
    scaling = np.sqrt(d_k)
    
    scaled_attention_score = raw_attention_score / scaling

    attention_score = softmax(scaled_attention_score)

    full_output = np.dot(attention_score, value)
    reduced_output = np.sum(full_output, axis=0)
    next_word = decode_output(reduced_output)
    print(next_word)
attention()






## forward prop:

## input layer * queryWeight, keyWeight, valueWeight (+bias)

## for each word:
## estimations = query * key = queryKey T * value

## softmax(value) = output


## back prop: 

## difference = 1/2(output - correct_output)^2

## gradient = aNeuron / aLayer 

## new_weight = old_weight - x()




## createMatrix()
## saveMatrix()
## loadMatrix()

## multiplyMatrix()


