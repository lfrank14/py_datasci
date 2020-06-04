##########################
###  IMPORT LIBRARIES  ###
##########################
import pandas as pd

from typing import TypeVar, Callable
narray = TypeVar('numpy.ndarray')
import numpy as np

import spacy
spnlp = TypeVar('spacy.lang.en.English')  #for type hints
import os
os.system("python -m spacy download en_core_web_md")


###################################
###  FUNCTIONS FOR NAIVE BAYES  ###
###################################




###########################
###  FUNCTIONS FOR ANN  ###
###########################

### Vector Math ###
def addv(x:list, y:list) -> list:
  assert isinstance(x, list), f"x must be a list but instead is {type(x)}"
  assert isinstance(y, list), f"y must be a list but instead is {type(y)}"
  assert len(x) == len(y), f"x and y must be the same length"

  result = [(c1+c2) for c1, c2 in zip(x,y)]
  return result

def dividev(x:list, y:int) -> list:
  assert isinstance(x, list), f"x must be a list but instead is {type(x)}"
  assert isinstance(y, int), f"y must be a list but instead is {type(y)}"

  result = [c/y for c in x]
  return result

def meanv(matrix:list) -> list:
  assert isinstance(matrix, list), f"x must be a list but instead is {type(x)}"
  assert len(matrix) >= 1, f"matrix must have at least one row"

  sumv = matrix[0]
  for row in matrix[1:]:
    sumv = addv(sumv,row)
  result = dividev(sumv, len(matrix))

  return result


### Get GloVe vector for a single string/word ### 
def get_vec(s:str) -> list:
    return nlp.vocab[s].vector.tolist()


### Get the mean GloVe vector for a single sentence ###
def sent2vec(sentence) -> list:
  assert isinstance(sentence, str), f'sentence is not a str but instead a {type(sentence)}'

  doc = nlp(sentence.lower())
  sentence_vectors = []
  for token in doc:
    if token.is_alpha and not token.is_stop:
      sentence_vectors.append(get_vec(token.text))
  
  if len(sentence_vectors) < 1:
    sentence_average = [0.0]*300
  else:
    sentence_average = meanv(sentence_vectors)

  return sentence_average
