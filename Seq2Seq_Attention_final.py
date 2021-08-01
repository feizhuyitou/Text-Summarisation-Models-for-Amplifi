#import spacy
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


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

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


class MyDataSet(Data.Dataset):
  def __init__(self, enc_inputs, dec_inputs, dec_outputs):
    super(MyDataSet, self).__init__()
    self.enc_inputs = enc_inputs
    self.dec_inputs = dec_inputs
    self.dec_outputs = dec_outputs

  def __len__(self):
    return self.enc_inputs.shape[0]

  def __getitem__(self, idx):
    return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

def np_to_tensor(obj):
    '''
    transfer the Series in dataframe to tensor
    obj: series or dataframe

    '''
    lis = obj.values.tolist()
    ten = torch.LongTensor(lis)
    return ten

# idx_data = pd.read_csv('idx_data_all.csv')
# X = idx_data['Content']#[:200]
# y = idx_data['Summary']#[:200]
# tgt = idx_data['Tgt_Summary']#[:200]
# X = X.apply(lambda x: literal_eval(x))
# y = y.apply(lambda x: literal_eval(x))
# tgt = tgt.apply(lambda x: literal_eval(x))
# X_train,X_val,y_train,y_val,tgt_train,tgt_val = train_test_split(X,y,tgt,test_size=0.1,random_state=12306)#split the data with random state 12306
# X_val,X_test,y_val,y_test,tgt_val,tgt_test = train_test_split(X_val,y_val,tgt_val,test_size=0.5,random_state=12306)#split the data with random state 12306
#
# # convert to dataset that can be accepted by torch
# enc_inputs, dec_inputs, dec_outputs = np_to_tensor(X_train), np_to_tensor(y_train), np_to_tensor(tgt_train)
# loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 20, True)
#
# dev_enc_inputs, dev_dec_inputs, dev_dec_outputs = np_to_tensor(X_val), np_to_tensor(y_val), np_to_tensor(tgt_val)
# dev_loader = Data.DataLoader(MyDataSet(dev_enc_inputs, dev_dec_inputs, dev_dec_outputs),20, False)
#
# test_enc_inputs, test_dec_inputs, test_dec_outputs = np_to_tensor(X_test), np_to_tensor(y_test), np_to_tensor(tgt_test)
# test_loader = Data.DataLoader(MyDataSet(test_enc_inputs, test_dec_inputs, test_dec_outputs),20, False)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        '''
        src = [src_len, batch_size]
        '''
        src = src.transpose(0, 1)  # src = [batch_size, src_len]
        embedded = self.dropout(self.embedding(src)).transpose(0, 1)  # embedded = [src_len, batch_size, emb_dim]

        # enc_output = [src_len, batch_size, hid_dim * num_directions]
        # enc_hidden = [n_layers * num_directions, batch_size, hid_dim]
        enc_output, enc_hidden = self.rnn(embedded)  # if h_0 is not give, it will be set 0 acquiescently

        # enc_hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # enc_output are always from the last layer

        # enc_hidden [-2, :, : ] is the last of the forwards RNN
        # enc_hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        # encoder RNNs fed through a linear layer
        # s = [batch_size, dec_hid_dim]
        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=1)))

        return enc_output, s


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]

        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, s, enc_output):
        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        dec_input = dec_input.unsqueeze(1)  # dec_input = [batch_size, 1]

        embedded = self.dropout(self.embedding(dec_input)).transpose(0, 1)  # embedded = [1, batch_size, emb_dim]

        # a = [batch_size, 1, src_len]
        a = self.attention(s, enc_output).unsqueeze(1)

        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        enc_output = enc_output.transpose(0, 1)

        # c = [1, batch_size, enc_hid_dim * 2]
        c = torch.bmm(a, enc_output).transpose(0, 1)

        # rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]
        rnn_input = torch.cat((embedded, c), dim=2)

        # dec_output = [src_len(=1), batch_size, dec_hid_dim]
        # dec_hidden = [n_layers * num_directions, batch_size, dec_hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0))

        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        embedded = embedded.squeeze(0)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)

        # pred = [batch_size, output_dim]
        pred = self.fc_out(torch.cat((dec_output, c, embedded), dim=1))

        return pred, dec_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.2):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        # teacher_forcing_ratio is probability to use teacher forcing

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output, s = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        dec_input = trg[0, :]

        for t in range(1, trg_len):
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            dec_output, s = self.decoder(dec_input, s, enc_output)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = dec_output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = dec_output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input = trg[t] if teacher_force else top1

        return outputs


INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(tgt_vocab)
ENC_EMB_DIM = 512
DEC_EMB_DIM = 512
ENC_HID_DIM = 1024
DEC_HID_DIM = 1024
ENC_DROPOUT = 0.0
DEC_DROPOUT = 0.0

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2Seq(enc, dec, device).to(device)
criterion = nn.CrossEntropyLoss(ignore_index = 1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


model.apply(init_weights)


def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    # for i, batch in tqdm(enumerate(iterator)):
    #     src = batch[0].T.cuda()
    #     trg = batch[1].T.cuda()  # trg = [trg_len, batch_size]
    for src, trg, out in tqdm(iterator):
        src = src.T.cuda()
        trg = trg.T.cuda()
        out = out.T.cuda()

        # pred = [trg_len, batch_size, pred_dim]
        pred = model(src, trg)

        pred_dim = pred.shape[-1]

        # trg = [(trg len - 1) * batch size]
        # pred = [(trg len - 1) * batch size, pred_dim]
        out = out[:-1].contiguous().view(-1)
        pred = pred[1:].contiguous().view(-1, pred_dim)

        loss = criterion(pred, out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


"""...and the evaluation loop, remembering to set the model to `eval` mode and turn off teaching forcing."""




def evaluate(model, iterator):
    model.eval()
    epoch_ROUGE = 0
    length = 0
    with torch.no_grad():
        for src_,_,out_ in iterator:
        # output = [trg_len, batch_size, output_dim]
            for idx in range(len(src_)):
                src = src_[idx].cuda()
                out = out_[idx].cuda()
                length += 1
                predict = greedy_decoder(model, src.view(-1, 1), start_symbol=tgt_vocab["<SoS>"])  # turn off teacher forcing
                predict = ' '.join([idx2word[n.item()] for n in predict.squeeze()])
                truth = ' '.join(['<SoS>'] + [idx2word[n.item()] for n in out.squeeze() if n != 1])
                ROUGE = rouge(predict, truth)[1]['r']
                epoch_ROUGE += ROUGE

    return epoch_ROUGE / length

def val_evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].T.cuda()
            trg = batch[1].T.cuda()  # trg = [trg_len, batch_size]
            out = batch[2].T.cuda()


            # output = [trg_len, batch_size, output_dim]
            output = model(src, trg, 0)  # turn off teacher forcing

            output_dim = output.shape[-1]

            # trg = [(trg_len - 1) * batch_size]
            # output = [(trg_len - 1) * batch_size, output_dim]
            output = output[1:].contiguous().view(-1, output_dim)
            # trg = trg[1:].contiguous().view(-1)
            out = out[:-1].contiguous().view(-1)

            loss = criterion(output, out)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


"""Finally, define a timing function."""


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def rouge(a, b):
    rouge = Rouge()
    rouge_score = rouge.get_scores(a, b, avg=True)  # a和b里面包含多个句子的时候用
    rouge_score1 = rouge.get_scores(a, b)  # a和b里面只包含一个句子的时候用
    # 以上两句可根据自己的需求来进行选择
    r1 = rouge_score["rouge-1"]
    r2 = rouge_score["rouge-2"]
    rl = rouge_score["rouge-l"]

    return r1, r2, rl


def greedy_decoder(model, enc_input, start_symbol):
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


best_valid_ROUGE = float('-inf')

# model.load_state_dict(torch.load('Att_Seq_last.pt'))

# enc_inputs, _, target = next(iter(dev_loader))
# enc_inputs = enc_inputs.cuda()
# ROU_list = []
# train_list = []
# val_list = []
# n_epochs = 50
# count = 0
#
# for epoch in range(n_epochs):
#     print('epoch {} starts'.format(epoch+1))
#     start_time = time.time()
#     count += 1
#
#     train_loss = train(model, loader, optimizer, criterion)
#     train_list.append(train_loss)
#     valid_loss = val_evaluate(model, dev_loader, criterion)
#     val_list.append(valid_loss)
#     # pred = []
#     # truth = []
#     # for i in range(len(enc_inputs)):
#     #     predict = greedy_decoder(model, enc_inputs[i].view(-1, 1), start_symbol=tgt_vocab["<SoS>"]).cuda()
#     #     pred.append(' '.join([idx2word[n.item()] for n in predict.squeeze()]))
#     #     truth.append(' '.join(['<SoS>'] + [idx2word[n.item()] for n in target[i].squeeze() if n != 1]))
#     #
#     # valid_ROUGE, _, _ = rouge(pred, truth)
#     # valid_ROUGE = valid_ROUGE['r']
#     valid_ROUGE = evaluate(model, dev_loader)
#     ROU_list.append(valid_ROUGE)
#
#     end_time = time.time()
#
#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
#
#     torch.save(model.state_dict(), 'Att_Seq_last.pt')
#
#     if valid_ROUGE > best_valid_ROUGE:
#         count = 0
#         print('Best, saved!')
#         best_valid_ROUGE = valid_ROUGE
#         torch.save(model.state_dict(), 'Att_Seq_best.pt')
#
#     print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f}')
#     print(f'\t Val. ROUGE_2: {valid_ROUGE:.3f}')
#
#     if count == 5:
#         print('early stop')
#         break

"""Finally, we test the model on the test set using these "best" parameters."""

def plot_train_vs_val_precision(label_list,hyp_list,train_list,val_list):
    '''
    plots the train vs. validation precision curve
    --- arguments ---
    label_list: a list with the format [title, X_label, y_label]
    hyp_list: list of values used for hyparameter optimisation (num-of-epochs in this case)
    train_precision: the list of precision score for training set
    val_precision: the list of precision score for testing set
    '''
    plt.figure(figsize=(10, 5))
    plt.title(label_list[0],fontsize=15,y=1.04)
    plt.plot(hyp_list,train_list, "b-", linewidth=1,label='train_set_loss')
    plt.plot(hyp_list,val_list, "g-", linewidth=1,label='validation_set_loss')
    plt.xlabel(label_list[1], fontsize=12)
    plt.ylabel(label_list[2], fontsize=12)
    plt.legend(loc="lower right", fontsize=8)
    plt.show()


# label_list = ['train_loss vs val_loss', 'n_epochs', 'loss']
# hyp_list = [i+1 for i in range(len(train_list))]
# plot_train_vs_val_precision(label_list,hyp_list,train_list,val_list)
#
# plt.figure(figsize=(10, 5))
# plt.title('ROUGE-2 Trend',fontsize=15,y=1.04)
# plt.plot(hyp_list, ROU_list, "r-", linewidth=1, label='validation_ROUGE-2')
# plt.xlabel(label_list[1], fontsize=12)
# plt.ylabel('ROUGE recall rate', fontsize=12)
# plt.legend(loc="lower right", fontsize=8)
# plt.show()
#
#
# model.load_state_dict(torch.load('Att_Seq_best.pt'))
#
# test_ROUGE = evaluate(model, test_loader)
#
# print(f'| Test ROUGE: {test_ROUGE:.3f}')







# # model.load_state_dict(torch.load('Att_Seq_last.pt'))
# enc_inputs, _, target = next(iter(test_loader))
# enc_inputs = enc_inputs.cuda()
# for i in range(len(enc_inputs)):
#     predict = greedy_decoder(model, enc_inputs[i].view(-1, 1), start_symbol=tgt_vocab["<SoS>"]).cuda()
#     # predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
#     # predict = predict.cuda()
#     # predict = predict.data.max(1, keepdim=True)[1]
#     print('Sentence {}'.format(i+1))
#     print('Raw Document: ',' '.join([idx2cont[j] for j in enc_inputs[i].cpu().numpy() if j!=1]),
#           '\n\n Generated Summary: ', ' '.join([idx2word[n.item()] for n in predict.squeeze()]),
#           '\n\n target: ', ' '.join(['<SoS>'] + [idx2word[n.item()] for n in target[i].squeeze() if n!=1]), '\n')
#     # print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])









