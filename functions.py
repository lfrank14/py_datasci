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
import en_core_web_md
nlp = en_core_web_md.load()

###################################
###  FUNCTIONS FOR NAIVE BAYES  ###
###################################

### Update word bag ###
def update_news_row(word_table, word:str, outcome:int):
  assert isinstance(outcome, int), f'Expecting int in outcome but saw {type(outcome)}.'
  value_list = [[1,0],[0,1]]
  word_list = word_table['word'].tolist()
  real_word = word if type(word) == str else word.text
  
  if real_word in word_list:
    j = word_list.index(real_word)
    word_table.loc[j, outcome] += 1
  else:
    row = [real_word] + value_list[outcome]
    word_table.loc[len(word_table)] = row
  return word_table

### Calculate naive bayes on test set ###
def bayes_news(evidence:list, evidence_bag, training_table, laplace:float=1.0) -> tuple:
  assert isinstance(evidence, list), f'evidence not a list but instead a {type(evidence)}'
  assert all([isinstance(item, str) for item in evidence]), f'evidence must be list of strings (not spacy tokens)'
  assert isinstance(evidence_bag, pd.core.frame.DataFrame), f'evidence_bag not a dataframe but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'

  label_list = training_table.isfake.to_list()
  word_list = evidence_bag.index.values.tolist()

  evidence = list(set(evidence))  #remove duplicates
  counts = []
  probs = []
  for i in range(2):
    ct = label_list.count(i)
    counts.append(ct)
    probs.append(ct/len(label_list))

  #now have counts and probs for all classes

  results = []
  for a_class in range(2):
    numerator = 1
    for ei in evidence:
      if ei not in word_list:
        #did not see word in training set
        the_value =  1/(counts[a_class] + len(evidence_bag))
      else:
        values = evidence_bag.loc[ei].tolist()
        the_value = ((values[a_class]+laplace)/(counts[a_class] + laplace*len(evidence_bag)))
      numerator *= the_value
    #if (numerator * probs[a_class]) == 0: print(evidence)
    results.append(max(numerator * probs[a_class], 2.2250738585072014e-308))

  return tuple(results)

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
