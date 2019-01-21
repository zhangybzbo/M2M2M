import os
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
from torch import optim
from model import Network, AttnNet, HiddenNet, TransformerNet
from unique_code import code_to_index, word_to_index, read_vocab, tokenizer, read_embed, pre_embed, get_elmo

Spell_check = False  # TODO: spell check not applied right
Pretrain_type = 'elmo_repre'  # bert / elmo_repre / elmo_layer

DATA_path = 'data/'
MODEL_path = 'models/'
word_list = 'wordls.txt'
code_list = 'codels.txt'
word_list_2 = 'wordls_2.txt'
code_list_2 = 'codels_2.txt'
char_list = 'charls.txt'
pretrained = 'Health_2.5mreviews.s200.w10.n5.v15.cbow.txt'

Max_seq_len = 40 if Pretrain_type == 'bert' else 35
HealthVec_size = 200
Embedding_size = 200 + 768 if Pretrain_type == 'bert' else 200 + 1024
Hidden_size = 200
Inner_hid_size = 1024
D_k = 64
D_v = 64
Num_layers = 6
Num_head = 5
Dropout = 0.2

Learning_rate = 0.0001
Weight_decay = 0.0015
LR_decay = 0.5
Epoch = 600
LR_decay_epoch = 300
Batch_size = 128

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def source_prepare():
    '''get vocabulary and HealthVec pre-train embedding'''
    code_vocab = read_vocab(DATA_path + code_list_2)
    code_id = code_to_index(code_vocab)
    word_vocab = read_vocab(DATA_path + word_list_2)
    word_id = word_to_index(word_vocab)
    # char_vocab = read_vocab(DATA_path + char_list)
    # char_id = word_to_index(char_vocab)

    raw_embedding, max_e, min_e = read_embed(MODEL_path + pretrained)
    if Spell_check:
        embeddings, vocab_correction = pre_embed(raw_embedding, word_vocab, max_e, min_e, HealthVec_size, Spell_check)
        embeddings = torch.tensor(embeddings)
    else:
        embeddings = pre_embed(raw_embedding, word_vocab, max_e, min_e, HealthVec_size, Spell_check)
        embeddings = torch.tensor(embeddings)

    return embeddings.cuda(), code_id, word_id


def data_prepare(code_id, word_id, train_file, test_file):
    '''prepare training and testing data'''
    train_data = tokenizer(word_id, code_id, train_file, pretrain_type=Pretrain_type)
    test_data = tokenizer(word_id, code_id, test_file, pretrain_type=Pretrain_type)

    return train_data, test_data


def adjust_learning_rate(optimizer, decay_rate=.5):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


if __name__ == "__main__":
    pretrain, code_id, word_id = source_prepare()
    print('%d different words, %d different codes' % (len(word_id), len(code_id)), flush=True)

    criterion = nn.CrossEntropyLoss()

    Acc = 0.
    # for fold in range(5):
    for fold in range(1):
        Net = TransformerNet(Pretrain_type, pretrain, Max_seq_len, Embedding_size, Inner_hid_size, len(code_id), D_k,
                             D_v, dropout_ratio=Dropout, num_layers=Num_layers, num_head=Num_head).cuda()
        optimizer = optim.Adam(Net.parameters(), lr=Learning_rate, eps=1e-08, weight_decay=Weight_decay)

        '''train_file = DATA_path + 'train_' + str(fold) + '.csv'
        test_file = DATA_path + 'test_' + str(fold) + '.csv'''
        train_file = DATA_path + 'train_final.txt'
        test_file = DATA_path + 'test_final.txt'
        train_data, test_data = data_prepare(code_id, word_id, train_file, test_file)

        print('Fold %d: %d training data, %d testing data' % (fold, len(train_data.data), len(test_data.data)),
              flush=True)

        for e in range(Epoch):
            train_data.reset_epoch()
            while not train_data.epoch_finish:
                Net.train()
                optimizer.zero_grad()
                seq, label, seq_length, mask, seq_pos, standard_emb = train_data.get_batch(Batch_size)
                print(seq.size(), standard_emb.size())
                results = Net(seq, seq_pos, standard_emb)
                loss = criterion(results, label)
                loss.backward()
                optimizer.step()

            Net.eval()

            train_data.reset_epoch()
            train_correct = 0
            while not train_data.epoch_finish:
                seq, label, seq_length, mask, seq_pos, standard_emb = test_data.get_batch(Batch_size)
                results = Net(seq, seq_pos, standard_emb)
                _, idx = results.max(1)
                train_correct += len((idx == label).nonzero())
            train_accuracy = float(train_correct) / float(len(train_data.data))

            test_data.reset_epoch()
            seq, label, seq_length, mask, seq_pos, standard_emb = test_data.get_batch(len(test_data.data))
            results = Net(seq, seq_pos, standard_emb)
            _, idx = results.max(1)
            test_correct = len((idx == label).nonzero())
            test_accuracy = float(test_correct) / float(len(test_data.data))

            if (e + 1) % 10 == 0:
                print('[fold %d epoch %d] training loss: %.4f, % d correct, %.4f accuracy;'
                      ' testing: %d correct, %.4f accuracy' %
                      (fold, e, loss.item(), train_correct, train_accuracy, test_correct, test_accuracy), flush=True)

            if (e + 1) % LR_decay_epoch == 0:
                adjust_learning_rate(optimizer, LR_decay)
                print('learning rate decay!', flush=True)

        Acc += test_accuracy

        del train_data, test_data
        gc.collect()

    print('finial test accuracy: %.4f' % (Acc / 1))
