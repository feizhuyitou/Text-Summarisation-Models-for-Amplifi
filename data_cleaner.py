import spacy
import numpy as np
import random
import math
import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import json
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ast import literal_eval
import pandas as pd
from rouge import Rouge
import matplotlib.pyplot as plt
import sklearn
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
from collections import Counter
from langdetect import detect

# data cleaning
import re
from zhon.hanzi import punctuation as ch_pun
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
from tqdm import tqdm
from sklearn.model_selection import train_test_split


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}


def clean_text(text, remove_stopwords=True):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''

    # Convert words to lower case
    text = text.lower()

    # Replace contractions with their longer forms
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]{}<>', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'•', '', text)
    text = re.sub(r'<strong>nbsp', '', text)
    text = re.sub('<p>', '', text)
    text = re.sub('<li>', '', text)
    text = re.sub('< li>', '', text)
    for i in list(ch_pun):
        text = re.sub(i, '', text)

    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text


try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    import spacy.cli

    print("Model not found. Downloading.")
    spacy.cli.download("en_core_web_md")
    import en_core_web_md

    nlp = en_core_web_md.load()


def spacy_word_tokenize(text):
    return [token.text for token in nlp(text)]


def drop_spaces(obj):
    '''
    remove the spaces which sometimes be tokenised.

    '''
    drop = obj.copy()
    for i in tqdm(range(len(drop))):
        drop[i] = [j for j in drop[i] if j != ' ' and j != '']
    return drop



def flat_list(obj):
    flatten = []
    for i in obj:
        for j in i:
            flatten.append(j)
    return flatten


def text_to_idx(token_obj, dic_obj):
    '''
    transfer tokens into indices.
    token_obj: list of tokens
    dic_obj: a dictionary with words as keys and count as values

    '''
    token = token_obj.copy()
    for idx in tqdm(range(len(token_obj))):
        token[idx] = [dic_obj[i] if i in src_vocab.keys() else 0 for i in token[idx]]
        # if the objective is not in the keys of the dict, we replace it with OOV token index 0
    return token


def add_SoS_EoS(idx_obj, SoS=True, EoS=True):
    '''
    add <SoS>, <EoS> to the tokenised sentences
    idx_obj: list of tokens that are transformed into indices

    '''
    obj = idx_obj.copy()
    for idx in tqdm(range(len(idx_obj))):
        if SoS:
            obj[idx] = [2] + obj[idx]
        if EoS:
            obj[idx] = obj[idx] + [3]
    return obj


def blank_filling(idx_obj, tgt=False, inputs=False):
    '''
    fill the sentences with <Blank> to ensure sentence length consistency
    idx_obj: list of tokens that are transformed into indices

    '''
    obj = idx_obj.copy()
    max_len = max([len(i) for i in obj])
    print('Maximum length is {}'.format(max_len))
    for idx in tqdm(range(len(obj))):
        if inputs == True:
            obj[idx] = obj[idx][:-1]
        current_len = len(obj[idx])
        obj[idx] = obj[idx] + [1] * (max_len - current_len)
        if tgt == True:
            obj[idx] = obj[idx][1:] + [1]
    print('Current length is {}'.format(len(obj[idx])))
    return obj



def clean(content,src_vocab,tgt = None, tgt_vocab = None):
    '''
    tokenise the input clause and transform the tokens into indices
    :param content: a list of clauses to be summarised
    :param src_vocab: dictionary of source document
    :param tgt: a list of target summaries
    :param tgt_vocab: dictionary of target summary document
    :param idx: True if the input is the cleaned version in indices
    :return: iterator of content and summary token lists
    '''
    content = [clean_text(i, remove_stopwords=False) for i in content]
    cont_tokens = [spacy_word_tokenize(i) for i in tqdm(content)]
    cont_tokens = drop_spaces(cont_tokens)
    cont_idx = text_to_idx(cont_tokens, src_vocab)
    cont_idx = [i[:297] for i in cont_idx]
    cont_idx = add_SoS_EoS(cont_idx)
    cont_idx = blank_filling(cont_idx)

    if tgt != None and tgt_vocab != None:
        tgt = [clean_text(i, remove_stopwords=False) for i in tgt]
        tgt_tokens = [spacy_word_tokenize(i) for i in tqdm(tgt)]
        tgt_tokens = drop_spaces(tgt_tokens)
        tgt_idx = text_to_idx(tgt_tokens, tgt_vocab)
        tgt_idx = [i[:27] for i in tgt_idx]
        tgt_idx = add_SoS_EoS(tgt_idx)
        tgt_idx = blank_filling(tgt_idx, inputs=True)
        tgt_idx = blank_filling(tgt_idx, tgt=True)
        idx_data = pd.DataFrame({'Content':cont_idx,'Summary':tgt_idx, 'Tgt_Summary':sum_idx_1})
        idx_data.to_csv('idx_data_all.csv', index=False)
    else:
        idx_data = pd.DataFrame({'Content':cont_idx})
        idx_data.to_csv('idx_data_all.csv', index=False)

def main():
    clean(content, src_vocab, tgt, tgt_vocab)

if __name__ == '__main__':
    root_dir = './'
    file_name = 'cont_w2i_all.json'
    str_file = root_dir + file_name
    with open(str_file, 'r') as f:
        print("Load str file from {}".format(str_file))
        src_vocab = json.load(f)
    idx2cont = {i: w for i, w in enumerate(src_vocab)}
    src_vocab_size = len(src_vocab)

    file_name = 'sum_w2i_all.json'
    str_file = root_dir + file_name
    with open(str_file, 'r') as f:
        print("Load str file from {}".format(str_file))
        tgt_vocab = json.load(f)
    idx2word = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    data_dir = './'
    data_name = 'idx_data_all.csv'
    data = pd.read_csv(data_dir)
    content = data['Content']
    tgt = None
    main()