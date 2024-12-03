from gensim.downloader import load
import numpy as np
import math 

## 300 dimensions TODO: Download locally/or database
glove_model = load('glove-wiki-gigaword-300')



## Temporary function until api
def embed_input():
    user_input = input("USER: ")
    sentence = user_input.split()
    embeddings = []

    for message in sentence:
        if message in glove_model:
            embeddings.append(glove_model[message])
        else: 
            print(f"Words could not be embedded") ##TODO Randomize here.
    return embeddings
embed_input()

def positional_encoding():



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


