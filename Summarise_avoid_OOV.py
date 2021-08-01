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


def np_to_tensor(obj):
    '''
    transfer the Series in dataframe to tensor
    obj: list of input

    '''
    ten = torch.LongTensor(obj)
    return ten


# convert to dataset that can be accepted by torch
class MyDataSet(Data.Dataset):
  def __init__(self, enc_inputs, dec_outputs):
    super(MyDataSet, self).__init__()
    self.enc_inputs = enc_inputs
    self.dec_outputs = dec_outputs

  def __len__(self):
    return self.enc_inputs.shape[0]

  def __getitem__(self, idx):
    return self.enc_inputs[idx], self.dec_outputs[idx]

def seq2seq_greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    s = model.encoder(enc_input)
    context = s
    dec_input = torch.zeros(0, 1).type_as(enc_input.data).cuda() #different from transformer -- (1,0) in tansformer
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input=torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype).cuda()],0).cuda()
        prob, s = model.decoder(dec_input[-1], s, context) # different from transformer
        # projected = model.projection(dec_outputs)
        # prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        prob = prob.max(dim=-1,keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        # if dec_input.size()[0] == source
        if next_symbol == tgt_vocab["<EoS>"] or len(dec_input) == 31:
            terminal = True
            dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype).cuda()], 0)
        # print(next_word)
    return dec_input

def attn_seq2seq_greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_output, s = model.encoder(enc_input)
    dec_input = torch.zeros(0, 1).type_as(enc_input.data).cuda() #different from transformer -- (1,0) in tansformer
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input=torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype).cuda()],0).cuda()
        prob, s = model.decoder(dec_input[-1], s, enc_output) # different from transformer
        # projected = model.projection(dec_outputs)
        # prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        prob = prob.max(dim=-1,keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        # if dec_input.size()[0] == source
        if next_symbol == tgt_vocab["<EoS>"] or len(dec_input) == 31:
            terminal = True
            dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype).cuda()], 0)
        # print(next_word)
    return dec_input

def transformer_greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data).cuda()
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input=torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype).cuda()],-1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["<EoS>"]: # or dec_input.shape(-1)==30:
            terminal = True
            dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype).cuda()], -1)
        # print(next_word)
    return dec_input

def clean(content,src_vocab,tgt = None, tgt_vocab = None, idx = False):
    '''
    tokenise the input clause and transform the tokens into indices
    :param content: a list of clauses to be summarised
    :param src_vocab: dictionary of source document
    :param tgt: a list of target summaries
    :param tgt_vocab: dictionary of target summary document
    :param idx: True if the input is the cleaned version in indices
    :return: iterator of content and summary token lists
    '''
    if not idx:
        content = [clean_text(i, remove_stopwords=False) for i in content]
        cont_tokens = [spacy_word_tokenize(i) for i in tqdm(content)]
        cont_tokens = drop_spaces(cont_tokens)
        cont_idx = text_to_idx(cont_tokens, src_vocab)
        cont_idx = add_SoS_EoS(cont_idx)
        cont_idx = blank_filling(cont_idx)

        if tgt != None and tgt_vocab != None:
            tgt = [clean_text(i, remove_stopwords=False) for i in tgt]
            tgt_tokens = [spacy_word_tokenize(i) for i in tqdm(tgt)]
            tgt_tokens = drop_spaces(tgt_tokens)
            tgt_idx = text_to_idx(tgt_tokens, tgt_vocab)
            tgt_idx = add_SoS_EoS(tgt_idx)
            tgt_idx = blank_filling(tgt_idx)
            loader = Data.DataLoader(MyDataSet(np_to_tensor(cont_idx), np_to_tensor(tgt_idx)), len(content), False)
        else:
            loader = Data.DataLoader(MyDataSet(np_to_tensor(cont_idx)), len(content), False)
    else:
        cont_idx = content
        if tgt != None and tgt_vocab != None:
            tgt_idx = tgt
            loader = Data.DataLoader(MyDataSet(np_to_tensor(cont_idx), np_to_tensor(tgt_idx)), len(content), False)
        else:
            loader = Data.DataLoader(MyDataSet(np_to_tensor(cont_idx)), len(content), False)
    return loader



def translate(iterator, trans_model = None, attn_model = None, seq2seq = None,
              trg = True, transformer = True, attention = False, traditional = False):
    '''
    Start translate raw contents into summaries and

    :param iterator: the combination of the cleaned content and target
    :param trans_model: transformer model
    :param attn_model: attention-based Seq2Seq model
    :param seq2seq: traditional Seq2Seq model
    :param trg: True if there is target summary in the iterator (turn it to False when trying the models on another dataset)
    :param transformer: activate the transformer translation (True by default since this is the best model)
    :param attention: activate the attention-based Seq2Seq model translation
    :param traditional: activate the traditional Seq2Seq model translation
    :return: return the cleaned raw content, cleaned target and the summaries generated by each model.

    '''
    if trg:
        enc_inputs, target = next(iter(iterator))
        enc_inputs = enc_inputs.cuda()
        target = target.cuda()

        for i in range(len(enc_inputs)):
            print('Sentence {}\n'.format(i + 1))
            print('Raw Document: ',' '.join([idx2cont[j] for j in enc_inputs[i].cpu().numpy() if j!=1]),
                '\n\n target: ', ' '.join(['<SoS>'] + [idx2word[n.item()] for n in target[i].squeeze() if n!=1]))
            if transformer and trans_model != None:
                predict = transformer_greedy_decoder(trans_model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["<SoS>"]).cuda()
                print('\n\n Transformer Summary: ', ' '.join([idx2word[n.item()] for n in predict.squeeze()]),'\n')
            if attention and attn_model != None:
                predict = attn_seq2seq_greedy_decoder(attn_model, enc_inputs[i].view(-1, 1), start_symbol=tgt_vocab["<SoS>"]).cuda()
                print('\n\n Attention based model Summary: ', ' '.join([idx2word[n.item()] for n in predict.squeeze()]),'\n')
            if traditional and seq2seq != None:
                predict = seq2seq_greedy_decoder(seq2seq, enc_inputs[i].view(-1, 1), start_symbol=tgt_vocab["<SoS>"]).cuda()
                print('\n\n traditional model Summary: ', ' '.join([idx2word[n.item()] for n in predict.squeeze()]), '\n')
    else:
        enc_inputs = next(iter(iterator))
        for i in range(len(enc_inputs)):
            print('Sentence {}\n'.format(i + 1))
            print('Raw Document: ', ' '.join(['<SoS>'] + [idx2cont[j] for j in enc_inputs[i].cpu().numpy() if j != 1]))
            if transformer and trans_model != None:
                predict = transformer_greedy_decoder(trans_model, enc_inputs[i].view(1, -1),
                                                     start_symbol=tgt_vocab["<SoS>"]).cuda()
                print('\n\n Transformer Summary: ', ' '.join([idx2word[n.item()] for n in predict.squeeze()]), '\n')
            if attention and attn_model != None:
                predict = attn_seq2seq_greedy_decoder(attn_model, enc_inputs[i].view(-1, 1),
                                                      start_symbol=tgt_vocab["<SoS>"]).cuda()
                print('\n\n Attention based model Summary: ', ' '.join([idx2word[n.item()] for n in predict.squeeze()]),
                      '\n')
            if traditional and seq2seq != None:
                predict = seq2seq_greedy_decoder(seq2seq, enc_inputs[i].view(-1, 1),
                                                 start_symbol=tgt_vocab["<SoS>"]).cuda()
                print('\n\n traditional model Summary: ', ' '.join([idx2word[n.item()] for n in predict.squeeze()]),
                      '\n')



def final_translate(iterator, trans_model, attn_model,trg = True):
    '''
    Start translate raw contents into summaries and

    :param iterator: the combination of the cleaned content and target
    :param trans_model: transformer model
    :param attn_model: attention-based Seq2Seq model
    :param trg: True if there is target summary in the iterator (turn it to False when trying the models on another dataset)
    :return: return the cleaned raw content, cleaned target and the summaries generated by each model.

    '''
    if trg:
        enc_inputs, target = next(iter(iterator))
        enc_inputs = enc_inputs.cuda()
        target = target.cuda()

        for i in range(len(enc_inputs)):
            print('Sentence {}\n'.format(i + 1))
            print('Raw Document: ',' '.join([idx2cont[j] for j in enc_inputs[i].cpu().numpy() if j!=1]),
                '\n\n target: ', ' '.join(['<SoS>'] + [idx2word[n.item()] for n in target[i].squeeze() if n!=1]))
            transformer_predict = transformer_greedy_decoder(trans_model, enc_inputs[i].view(1, -1),
                                                             start_symbol=tgt_vocab["<SoS>"]).cuda()
            transformer_predict = [n.item() for n in transformer_predict.squeeze()]
            if 0 in transformer_predict or 512 in transformer_predict: # Avoid OOV!!!
                predict = attn_seq2seq_greedy_decoder(attn_model, enc_inputs[i].view(-1, 1),
                                                      start_symbol=tgt_vocab["<SoS>"]).cuda()
                print('\n\n Attention based model Summary: ', ' '.join([idx2word[n.item()] for n in predict.squeeze()]),
                      '\n')

            else:
                print('\n\n Transformer Summary: ', ' '.join([idx2word[n] for n in transformer_predict]),'\n')
    else:
        enc_inputs = next(iter(iterator))
        for i in range(len(enc_inputs)):
            print('Sentence {}\n'.format(i + 1))
            print('Raw Document: ', ' '.join(['<SoS>'] + [idx2cont[j] for j in enc_inputs[i].cpu().numpy() if j != 1]))
            transformer_predict = transformer_greedy_decoder(trans_model, enc_inputs[i].view(1, -1),
                                                             start_symbol=tgt_vocab["<SoS>"]).cuda()
            transformer_predict = [n.item() for n in transformer_predict.squeeze()]
            if 0 in transformer_predict or 512 in transformer_predict: # Avoid OOV!!!
                predict = attn_seq2seq_greedy_decoder(attn_model, enc_inputs[i].view(-1, 1),
                                                      start_symbol=tgt_vocab["<SoS>"]).cuda()
                print('\n\n Attention based model Summary: ', ' '.join([idx2word[n.item()] for n in predict.squeeze()]),
                      '\n')

            else:
                print('\n\n Transformer Summary: ', ' '.join([idx2word[n] for n in transformer_predict]), '\n')



def main():
    iterator = clean(content, src_vocab, tgt, tgt_vocab, idx)
    final_translate(iterator, trans_model, attn_model, trg)


if __name__ == "__main__":
    from Transformer_Torch_with_gpu_final import Transformer
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

    idx_data = pd.read_csv('idx_data_all.csv')
    X = idx_data['Content']  # [:200]
    y = idx_data['Summary']  # [:200]
    y = idx_data['Content']
    tgt = idx_data['Tgt_Summary']  # [:200]
    X = X.apply(lambda x: literal_eval(x))
    y = y.apply(lambda x: literal_eval(x))
    tgt = tgt.apply(lambda x: literal_eval(x))
    X_train, X_val, y_train, y_val, tgt_train, tgt_val = train_test_split(X, y, tgt, test_size=0.1,
                                                                          random_state=12306)  # split the data with random state 12306
    X_val, X_test, y_val, y_test, tgt_val, tgt_test = train_test_split(X_val, y_val, tgt_val, test_size=0.5,
                                                                       random_state=12306)  # split the data with random state 12306

    # convert to dataset that can be accepted by torch
    content = X_test[:20].tolist()
    tgt = tgt_test[:20].tolist()
    trans_model = Transformer().cuda()
    trans_model.load_state_dict(torch.load('transformer_best.pt'))
    # attn_model = None
    # seq2seq = None
    idx = True
    trg = True

    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(tgt_vocab)
    ENC_EMB_DIM = 512
    DEC_EMB_DIM = 512
    ENC_HID_DIM = 1024
    DEC_HID_DIM = 1024
    ENC_DROPOUT = 0.0
    DEC_DROPOUT = 0.0

    from Seq2Seq_Attention_final import Attention, Encoder, Decoder, Seq2Seq
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    attn_model = Seq2Seq(enc, dec, device).to(device)
    # attn_model.load_state_dict(torch.load('Att_Seq_default.pt'))
    attn_model.load_state_dict(torch.load('Att_Seq_refined.pt'))

    main()












