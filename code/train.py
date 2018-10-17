import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from unique_code import code_to_index, word_to_index, read_vocab, tokenizer, read_embed, pre_embed

DATA_path = 'data/'
MODEL_path = 'model/'
word_list = 'wordls.txt'
code_list = 'codels.txt'
pretrained = 'Health_2.5mreviews.s200.w10.n5.v15.cbow.txt'
#train_file = 'train_0.csv'
#test_file = 'test_0.csv'

Embedding_size = 200
Hidden_size = 200
Learning_rate = 0.0005
Epoch = 100
Batch_size = 128

# without attention
class Network(nn.Module):
    def __init__(self, pretrained_embed, embedding_size, hidden_size, output_size, dropout_ratio=0.5, num_layers=1, bidirectional=True):
        super(Network, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.out = nn.Linear(hidden_size*self.num_directions, output_size)
        self.BN = nn.BatchNorm1d(output_size) # size?
        #self.softmax = nn.LogSoftmax(dim=1)
        
    def init_state(self, inputs):
        # input size (seq, batch)
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return h0.cuda(), c0.cuda()
        
    def forward(self, inputs):
        embeds = self.embedding(inputs)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        ctx, (h_t, c_t) = self.lstm(embeds, (h0, c0))
        ctx = torch.mean(ctx, 1)
        ctx_out = nn.Tanh()(self.out(ctx))
        output = self.BN(ctx_out)
        #output = self.softmax(output)
        return output

# unfinished
class AttnNet(nn.Module):
    def __init__(self, pretrained_embed, embedding_size, hidden_size, output_size, dropout_ratio=0.5, bidirectional=True):
        super(AttnNet, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding.from_pretrained(pretrained_embed)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.out = nn.Linear(hidden_size*self.num_directions, output_size)
        self.BN = nn.BatchNorm1d(output_size) # size?
        self.softmax = nn.LogSoftmax(dim=1)
        
    def init_state(self, inputs):
        # input size (seq, batch)
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return h0.cuda(), c0.cuda()
        
    def forward(self, inputs):
        embeds = self.embedding(inputs)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        
        ctx, (h_t, c_t) = self.lstm(embeds, (h0, c0))
        ctx = torch.mean(ctx, 1)
        ctx_out = nn.Tanh(self.out(ctx))
        ctx_out = self.BN(ctx_out)
        output = self.softmax(ctx_out)
   

def source_prepare():
    code_vocab = read_vocab(DATA_path + code_list)
    code_id = code_to_index(code_vocab)
    word_vocab = read_vocab(DATA_path + word_list)
    word_id = word_to_index(word_vocab)
    
    raw_embedding, max_e, min_e = read_embed(MODEL_path + pretrained)
    embeddings = pre_embed(raw_embedding, word_vocab, max_e, min_e, Embedding_size)
    
    return embeddings, code_id, word_id

def data_prepare(code_id, word_id, train_file, test_file):
    train_data = tokenizer(word_id, code_id, train_file)
    test_data = tokenizer(word_id, code_id, test_file)
    
    return train_data, test_data
    
if __name__ == "__main__":
    pretrain, code_id, word_id = source_prepare()
    print('%d different words, %d different codes' % (len(word_id), len(code_id)))
        
    
    criterion = nn.CrossEntropyLoss()
    
    Acc = 0.
    for fold in range(5):
        Net = Network(torch.tensor(pretrain), Embedding_size, Hidden_size, len(code_id)).cuda()    
        optimizer = optim.Adam(Net.parameters(), lr=Learning_rate)
        
        train_file = DATA_path + 'train_' + str(fold) + '.csv'
        test_file = DATA_path + 'test_' + str(fold) + '.csv'
        train_data, test_data = data_prepare(code_id, word_id, train_file, test_file)
        print('Fold %d: %d training data, %d testing data' % (fold, len(train_data.data), len(test_data.data)))
        
        for e in range(Epoch):
            train_data.reset_epoch()
            while not train_data.epoch_finish:
                Net.train()
                optimizer.zero_grad()
                seq, label, seq_length, mask = train_data.get_batch(Batch_size)
                results = Net(seq)
                loss = criterion(results, label)
                loss.backward()
                optimizer.step()
                    
            test_data.reset_epoch()
            Net.eval()
            seq, label, seq_length, mask = test_data.get_batch(len(test_data.data))
            results = Net(seq)
            _, idx = results.max(1)
            correct = len((idx == label).nonzero())    
            accuracy = float(correct) / float(len(test_data.data))
            
            if (e+1) % 10 == 0:
                print('[fold %d epoch %d] training loss: %.4f, testing: %d correct, %.4f accuracy' % (fold, e, loss.item(), correct, accuracy))
        
        Acc += accuracy
        
    print('finial accuracy: %.4f' % (Acc/5))
        
        