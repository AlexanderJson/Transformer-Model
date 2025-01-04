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
                embeddings.append(np.random.rand(dims))
        return np.array(embeddings)


def positional_encoding(input, dims):
    position = np.arange(input)[:, np.newaxis]
    div_term = np.exp(np.arange(0, dims, 2) * -(math.log(10000.0) / dims))

    PE = np.zeros((input, dims))
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)
    return PE

## UTIL functions

def softmax(matrix):
    exp = np.exp(matrix - np.max(matrix))
    return exp / np.sum(exp, axis=  -1, keepdims=True)

def masked_input(matrix):
    rows, columns = matrix.shape
    if rows != columns:
        raise ValueError("Invalid format for matrix")
    return np.triu(np.ones((rows,columns)), k=1) * -1e9

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
    masked_scores = masked_input(scaled_attention_score)
    attention_score = softmax(masked_scores)
    return np.dot(attention_score, value)

def predict_next_word(attention_output):
    logits = np.dot(attention_output, random_matrix(cols,dims))
    return softmax(logits)


def decoder():
    input_matrix = encode_input()
    input_length = input_matrix.shape[0]

    pe_matrix = positional_encoding(input_length, dims)
    input_with_pe = input_matrix + pe_matrix

    ## PE here
    attention_matrix = attention(input_with_pe)
    upscaled_input = FFN(attention_matrix)
    residual_matrix = residual_connection(input_with_pe, upscaled_input)
    normalised_output = normalise_matrix(residual_matrix)
    next_token_probability = predict_next_word(normalised_output)
    output_vector = next_token_probability.mean(axis=0)
    output_vector /= np.linalg.norm(output_vector)
    closest_word = glove_model.similar_by_vector(output_vector, topn=1)[0][0]
    print(closest_word)
decoder()


    


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