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

idx_data = pd.read_csv('idx_data_all.csv')
X = idx_data['Content']#[:200]
y = idx_data['Summary']#[:200]
tgt = idx_data['Tgt_Summary']#[:200]
X = X.apply(lambda x: literal_eval(x))
y = y.apply(lambda x: literal_eval(x))
tgt = tgt.apply(lambda x: literal_eval(x))
X_train,X_val,y_train,y_val,tgt_train,tgt_val = train_test_split(X,y,tgt,test_size=0.1,random_state=12306)#split the data with random state 12306
X_val,X_test,y_val,y_test,tgt_val,tgt_test = train_test_split(X_val,y_val,tgt_val,test_size=0.5,random_state=12306)#split the data with random state 12306

# convert to dataset that can be accepted by torch
enc_inputs, dec_inputs, dec_outputs = np_to_tensor(X_train), np_to_tensor(y_train), np_to_tensor(tgt_train)
loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 20, True)

dev_enc_inputs, dev_dec_inputs, dev_dec_outputs = np_to_tensor(X_val), np_to_tensor(y_val), np_to_tensor(tgt_val)
dev_loader = Data.DataLoader(MyDataSet(dev_enc_inputs, dev_dec_inputs, dev_dec_outputs),20, False)

test_enc_inputs, test_dec_inputs, test_dec_outputs = np_to_tensor(X_test), np_to_tensor(y_test), np_to_tensor(tgt_test)
test_loader = Data.DataLoader(MyDataSet(test_enc_inputs, test_dec_inputs, test_dec_outputs),20, False)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)  # no dropout as only one layer!

        self.rnn = nn.GRU(emb_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)  # no cell state!

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # context = [n layers * n directions, batch size, hid dim]

        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        emb_con = torch.cat((embedded, context), dim=2)

        # emb_con = [1, batch size, emb dim + hid dim]

        output, hidden = self.rnn(emb_con, hidden)

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # seq len, n layers and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)),
                           dim=1)

        # output = [batch size, emb dim + hid dim * 2]

        prediction = self.fc_out(output)

        # prediction = [batch size, output dim]

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is the context
        context = self.encoder(src)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(tgt_vocab)
ENC_EMB_DIM = 512
DEC_EMB_DIM = 512
HID_DIM = 1024
ENC_DROPOUT = 0.0
DEC_DROPOUT = 0.0


enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


model.apply(init_weights)



def train(model, iterator, optimizer, criterion, clip = None):
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
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

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
    rouge_score = rouge.get_scores(a, b, avg=True)  # a???b????????????????????????????????????
    rouge_score1 = rouge.get_scores(a, b)  # a???b???????????????????????????????????????
    # ???????????????????????????????????????????????????
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


best_valid_ROUGE = float('-inf')

# model.load_state_dict(torch.load('Att_Seq_last.pt'))

# enc_inputs, _, target = next(iter(dev_loader))
# enc_inputs = enc_inputs.cuda()
# ROU_list = []
# train_list = []
# val_list = []
# n_epochs = 50
# count = 0
# CLIP = 1
# optimizer = optim.Adam(model.parameters())
# # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
# criterion = nn.CrossEntropyLoss(ignore_index = 1)
# model.load_state_dict(torch.load('Seq_last.pt'))
#
# for epoch in range(n_epochs):
#     print('epoch {} starts'.format(epoch+1))
#     start_time = time.time()
#     count += 1
#
#     train_loss = train(model, loader, optimizer, criterion, CLIP)
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
#     torch.save(model.state_dict(), 'Seq_last.pt')
#
#     if valid_ROUGE > best_valid_ROUGE:
#         count = 0
#         print('Best, saved!')
#         best_valid_ROUGE = valid_ROUGE
#         torch.save(model.state_dict(), 'Seq_best.pt')
#
#     print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f}')
#     print(f'\t Val. ROUGE_2: {valid_ROUGE:.3f}')
#
#     if count == 50:
#         print('early stop')
#         break
#
# """Finally, we test the model on the test set using these "best" parameters."""
#
# def plot_train_vs_val_precision(label_list,hyp_list,train_list,val_list):
#     '''
#     plots the train vs. validation precision curve
#     --- arguments ---
#     label_list: a list with the format [title, X_label, y_label]
#     hyp_list: list of values used for hyparameter optimisation (num-of-epochs in this case)
#     train_precision: the list of precision score for training set
#     val_precision: the list of precision score for testing set
#     '''
#     plt.figure(figsize=(10, 5))
#     plt.title(label_list[0],fontsize=15,y=1.04)
#     plt.plot(hyp_list,train_list, "b-", linewidth=1,label='train_set_loss')
#     plt.plot(hyp_list,val_list, "g-", linewidth=1,label='validation_set_loss')
#     plt.xlabel(label_list[1], fontsize=12)
#     plt.ylabel(label_list[2], fontsize=12)
#     plt.legend(loc="lower right", fontsize=8)
#     plt.show()
#
#
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
# model.load_state_dict(torch.load('Seq_best.pt'))
#
# test_ROUGE = evaluate(model, test_loader)
#
# print(f'| Test ROUGE: {test_ROUGE:.3f}')



# # test some of the summaries
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










