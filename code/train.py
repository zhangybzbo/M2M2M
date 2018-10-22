import torch
import torch.nn as nn
from torch import optim
from model import Network, AttnNet, HiddenNet, TransformerNet
from unique_code import code_to_index, word_to_index, read_vocab, tokenizer, read_embed, pre_embed

DATA_path = 'data/'
MODEL_path = 'models/'
word_list = 'wordls.txt'
code_list = 'codels.txt'
pretrained = 'Health_2.5mreviews.s200.w10.n5.v15.cbow.txt'
# train_file = 'train_0.csv'
# test_file = 'test_0.csv'

Max_seq_len = 35
Embedding_size = 200
Hidden_size = 200
Inner_hid_size = 2048
D_k = 64
D_v = 64

Learning_rate = 0.0001
Weight_decay = 0.0015
LR_decay = 0.5
Epoch = 500
LR_decay_epoch = 200
Batch_size = 128


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


def adjust_learning_rate(optimizer, decay_rate=.5):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


if __name__ == "__main__":
    pretrain, code_id, word_id = source_prepare()
    print('%d different words, %d different codes' % (len(word_id), len(code_id)))

    criterion = nn.CrossEntropyLoss()

    Acc = 0.
    for fold in range(5):
        '''Net = TransformerNet(torch.tensor(pretrain), Max_seq_len, Embedding_size, Inner_hid_size, len(code_id), D_k,
                             D_v).cuda()'''
        Net = AttnNet(torch.tensor(pretrain), Embedding_size, Hidden_size, len(code_id)).cuda()
        optimizer = optim.Adam(Net.parameters(), lr=Learning_rate, weight_decay=Weight_decay)

        train_file = DATA_path + 'train_' + str(fold) + '.csv'
        test_file = DATA_path + 'test_' + str(fold) + '.csv'
        train_data, test_data = data_prepare(code_id, word_id, train_file, test_file)
        # max_seq_len = max(train_data.max_length, test_data.max_length)
        print('Fold %d: %d training data, %d testing data' % (fold, len(train_data.data), len(test_data.data)))
        # print('Max sequence length: %d' % max_seq_len)

        for e in range(Epoch):
            train_data.reset_epoch()
            while not train_data.epoch_finish:
                Net.train()
                optimizer.zero_grad()
                seq, label, seq_length, mask, seq_pos = train_data.get_batch(Batch_size)
                # results = Net(seq, seq_pos)
                results = Net(seq, mask)
                loss = criterion(results, label)
                loss.backward()
                optimizer.step()

            test_data.reset_epoch()
            Net.eval()
            seq, label, seq_length, mask, seq_pos = test_data.get_batch(len(test_data.data))
            # results = Net(seq, seq_pos)
            results = Net(seq, mask)
            _, idx = results.max(1)
            correct = len((idx == label).nonzero())
            accuracy = float(correct) / float(len(test_data.data))

            if (e + 1) % 10 == 0:
                print('[fold %d epoch %d] training loss: %.4f, testing: %d correct, %.4f accuracy' % (
                    fold, e, loss.item(), correct, accuracy))

            if (e + 1) % LR_decay_epoch == 0:
                adjust_learning_rate(optimizer, LR_decay)
                print('learning rate decay!')

        Acc += accuracy

    print('finial accuracy: %.4f' % (Acc / 5))
