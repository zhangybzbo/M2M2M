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

DATA_path = 'datasets/'
MODEL_path = 'models/'
SAVE_DIR = 'models/Normal/'
word_list = 'wordls.txt'
code_list = 'codels.txt'
char_list = 'charls.txt'
pretrained = 'Health_2.5mreviews.s200.w10.n5.v15.cbow.txt'

Freeze_emb = False
Max_seq_len = 40 if Pretrain_type == 'bert' else 35
HealthVec_size = 200
Embedding_size = 200 + 768 if Pretrain_type == 'bert' else 200 + 1024
Hidden_size = 200
Inner_hid_size = 1024
D_k = 64
D_v = 64
Num_layers = 6
Num_head = 5
Dropout = 0.3

Learning_rate = 0.0001
Weight_decay = 0.0015
LR_decay = 0.5
Epoch = 300
LR_decay_epoch = 300
Batch_size = 100
Val_every = 20

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def source_prepare():
    '''get vocabulary and HealthVec pre-train embedding'''
    code_vocab = read_vocab(DATA_path + code_list)
    code_id = code_to_index(code_vocab)
    word_vocab = read_vocab(DATA_path + word_list)
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


def adjust_learning_rate(optimizer, decay_rate=.5):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def train():
    pretrain, code_id, word_id = source_prepare()
    print('%d different words, %d different codes' % (len(word_id), len(code_id)), flush=True)

    criterion = nn.CrossEntropyLoss()

    Acc = 0.
    for fold in range(10):
        Net = TransformerNet(Pretrain_type, pretrain, Max_seq_len, Embedding_size, Inner_hid_size, len(code_id), D_k,
                             D_v, dropout_ratio=Dropout, num_layers=Num_layers, num_head=Num_head, Freeze=Freeze_emb).cuda()
        optimizer = optim.Adam(Net.parameters(), lr=Learning_rate, eps=1e-08, weight_decay=Weight_decay)

        train_file = DATA_path + 'AskAPatient/AskAPatient.fold-' + str(fold) + '.train.txt'
        val_file = DATA_path + 'AskAPatient/AskAPatient.fold-' + str(fold) + '.validation.txt'
        train_data = tokenizer(word_id, code_id, train_file, pretrain_type=Pretrain_type)
        val_data = tokenizer(word_id, code_id, val_file, pretrain_type=Pretrain_type)

        print('Fold %d: %d training data, %d validation data' % (fold, len(train_data.data), len(val_data.data)))
        print('max length: %d %d' % (train_data.max_length, val_data.max_length), flush=True)

        for e in range(Epoch):
            train_data.reset_epoch()
            Net.train()
            while not train_data.epoch_finish:
                optimizer.zero_grad()
                seq, label, seq_length, mask, seq_pos, standard_emb = train_data.get_batch(Batch_size)
                results = Net(seq, seq_pos, standard_emb)
                loss = criterion(results, label)
                loss.backward()
                optimizer.step()

            if (e + 1) % Val_every == 0:
                Net.eval()

                train_data.reset_epoch()
                train_correct = 0
                i = 0
                while not train_data.epoch_finish:
                    seq, label, seq_length, mask, seq_pos, standard_emb = train_data.get_batch(Batch_size)
                    results = Net(seq, seq_pos, standard_emb)
                    _, idx = results.max(1)
                    train_correct += len((idx == label).nonzero())
                    i += Batch_size
                # assert i == len(train_data.data)
                train_accuracy = float(train_correct) / float(i)

                val_data.reset_epoch()
                val_correct = 0
                i = 0
                while not val_data.epoch_finish:
                    seq, label, seq_length, mask, seq_pos, standard_emb = val_data.get_batch(Batch_size)
                    results = Net(seq, seq_pos, standard_emb)
                    _, idx = results.max(1)
                    val_correct += len((idx == label).nonzero())
                    i += Batch_size
                # assert i == len(val_data.data)
                val_accuracy = float(val_correct) / float(len(val_data.data))

                print('[fold %d epoch %d] training loss: %.4f, % d correct, %.4f accuracy;'
                      ' validation: %d correct, %.4f accuracy' %
                      (fold, e, loss.item(), train_correct, train_accuracy, val_correct, val_accuracy), flush=True)

                torch.save(Net.state_dict(), SAVE_DIR + 'Net_false_' + str(fold) + '_' + str(e))

            if (e + 1) % LR_decay_epoch == 0:
                adjust_learning_rate(optimizer, LR_decay)
                print('learning rate decay!', flush=True)

        Acc += val_accuracy

        del train_data, val_data
        gc.collect()

    print('finial validation accuracy: %.4f' % (Acc / 10))

def test():
    pretrain, code_id, word_id = source_prepare()
    print('%d different words, %d different codes' % (len(word_id), len(code_id)), flush=True)

    Acc = 0.
    for fold in range(10):
        Net = TransformerNet(Pretrain_type, pretrain, Max_seq_len, Embedding_size, Inner_hid_size, len(code_id), D_k,
                             D_v, dropout_ratio=Dropout, num_layers=Num_layers, num_head=Num_head, Freeze=Freeze_emb).cuda()
        Net.load_state_dict(torch.load(SAVE_DIR + 'Net_false_' + str(fold) + '_299'))
        Net.eval()

        test_file = DATA_path + 'AskAPatient/AskAPatient.fold-' + str(fold) + '.test.txt'
        test_data = tokenizer(word_id, code_id, test_file, pretrain_type=Pretrain_type)

        print('Fold %d: %d test data' % (fold, len(test_data.data)))
        print('max length: %d' % test_data.max_length, flush=True)

        test_data.reset_epoch()
        test_correct = 0
        i = 0
        while not test_data.epoch_finish:
            seq, label, seq_length, mask, seq_pos, standard_emb = test_data.get_batch(1)
            results = Net(seq, seq_pos, standard_emb)
            _, idx = results.max(1)
            test_correct += len((idx == label).nonzero())
            i += 1
        assert i == len(test_data.data)
        test_accuracy = float(test_correct) / float(i)

        print('[fold %d] test: %d correct, %.4f accuracy' % (fold, test_correct, test_accuracy), flush=True)

        Acc += test_accuracy

        del test_data
        gc.collect()

    print('finial validation accuracy: %.4f' % (Acc / 10), flush=True)


if __name__ == "__main__":
    train()
    test()