from gensim.downloader import load
import numpy as np
import math 

dims = 300
cols = 300

## 300 dimensions TODO: Download locally/or database
glove_model = load(f'glove-wiki-gigaword-{dims}')

### ENCODER ###


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

def FFN(matrix):
    ## ScaleUp Weight + Bias
    S_W = random_matrix(dims,cols)
    S_B = random_matrix(1, cols)
    upscaled =  np.dot(matrix, S_W) + S_B
    ReLU_activation =  np.maximum(0, upscaled)
    ## Compress Weight + Bias
    C_W = random_matrix(dims,cols)
    C_B = random_matrix(1, cols)
    return np.dot(ReLU_activation, C_W) + C_B

def residual_connection(input_matrix, transformed_output):
    return input_matrix + transformed_output

def normalise_matrix(matrix, epsilon=1e-6):
    mean = np.mean(matrix, axis=-1, keepdims=True)
    variance = np.var(matrix, axis=-1, keepdims=True)
    normalised_matrix = (matrix - mean) / np.sqrt(variance + epsilon)
    gamma = np.ones(matrix.shape[-1])
    beta = np.zeros(matrix.shape[-1])
    return gamma * normalised_matrix + beta

def attention(input_matrix):
    
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

    return np.dot(attention_score, value)


def encoder():
    input_matrix = encode_input()
    ## PE here
    attention = attention(input_matrix)
    upscaled_input = FFN(attention)
    residual_connection = residual_connection(upscaled_input)
    return normalise_matrix(residual_connection)


    
    


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