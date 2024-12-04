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
    print(full_output)
    return full_output
attention()






## back prop: 

## difference = 1/2(output - correct_output)^2

## gradient = aNeuron / aLayer 

## new_weight = old_weight - x()




## createMatrix()
## saveMatrix()
## loadMatrix()

## multiplyMatrix()




## NOTE: Must find a way to make AI write a proper response but know when to stop. Perhaps 
## relational treaining? Question -> Answer ?