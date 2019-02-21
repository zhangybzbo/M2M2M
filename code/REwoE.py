import os
import csv
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from model import SeqLayer, RelationDetect_woemb
from utils import dir_reader, relation_reader, tokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
epsilon = sys.float_info.epsilon

SAVE_DIR = 'models/snapshots/'
LOG_FILE = 'models/val.csv'
TRAIN_DIR = 'corpus/train/'
TEST_DIR = 'corpus/test/'
RELATIONS = 'data/relations.txt'

Relation_threshold = [0.3, 0.4, 0.5]
Relation_type = 10
Max_seq_len = 100
ELMo_size = 1024
Hidden_size = 100
Hidden_layer = 3
Dropout = 0.5
Bidirection = True

Learning_rate = 0.0001
LR_decay = 10
Weight_decay = 0.0005
Epoch = 500
Batch_size = 50
Val_every = 20
Log_every = 20


def get_REteacher(s, length, e_posi, relation):
    teacher = []
    count = []
    for i in range(Batch_size):
        if s - 1 in e_posi[i][1] and e_posi[i][1][0] > e_posi[i][0][0]:
            tmp = [p * Relation_type + relation[i].item() for p in e_posi[i][0]]
        elif s - 1 in e_posi[i][0] and e_posi[i][0][0] > e_posi[i][1][0]:
            tmp = [p * Relation_type + relation[i].item() for p in e_posi[i][1]]
        elif s > length[i]:
            tmp = [-1]
        else:
            tmp = [(s - 1) * Relation_type]
        count.append(len(tmp))
        teacher.append(tmp)
    return teacher, count


def end2end(train_data, val_data, LSTM_layer, RE, lr, epoch):
    RE_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    LSTM_optimizer = optim.Adam(LSTM_layer.parameters(), lr=lr/LR_decay, weight_decay=Weight_decay)
    RE_optimizer = optim.Adam(RE.parameters(), lr=lr, weight_decay=Weight_decay)

    for e in range(epoch):
        train_data.reset_epoch()
        LSTM_layer.train()
        RE.train()
        RE_losses = []
        while not train_data.epoch_finish:
            LSTM_optimizer.zero_grad()
            RE_optimizer.zero_grad()
            standard_emb, e_label, e_posi, r_label, seq_length, mask, seq_pos = train_data.get_batch(Batch_size)
            #print(standard_emb.size())
            #print(e_label)
            #print(e_posi, r_label, seq_length)
            #input()

            ctx = LSTM_layer(standard_emb, seq_length)
            #print(ctx.size())
            #input()
            relation_loss = 0.
            # relationship
            for s in range(1, max(seq_length) + 1):  # s is the count of word number
                u = RE(ctx[:, :s, :])
                #print(u.size())
                re_teacher, re_count = get_REteacher(s, seq_length, e_posi, r_label)
                #print(re_teacher, re_count)
                for i in range(Batch_size):
                    for j in range(re_count[i]):
                        teacher_ij = torch.tensor(re_teacher[i][j], dtype=torch.long, requires_grad=False).cuda()
                        relation_loss += RE_criterion(u[i, :, :].view(1, -1), teacher_ij.view(1)) / re_count[i]
                        #print(i,j,teacher_ij,relation_loss)
                        #input()

            relation_loss.backward()
            LSTM_optimizer.step()
            RE_optimizer.step()
            RE_losses.append(relation_loss.item())


        if (e + 1) % Val_every == 0:
            val_data.reset_epoch()
            LSTM_layer.eval()
            RE.eval()
            TP = [[0.] * Relation_type] * len(Relation_threshold)
            FP = [[0.] * Relation_type] * len(Relation_threshold)
            FN = [[0.] * Relation_type] * len(Relation_threshold)
            F1 = [[0.] * Relation_type] * len(Relation_threshold)
            total_F1 = [0.] * len(Relation_threshold)
            micro_F1 = [0.] * len(Relation_threshold)
            while not val_data.epoch_finish:
                standard_emb, e_label, e_posi, r_label, seq_length, mask, seq_pos = val_data.get_batch(Batch_size)
                #print(standard_emb.size())
                #print(e_label)
                #print(e_posi, r_label, seq_length)
                #input()
                ctx = LSTM_layer(standard_emb, seq_length)

                # get relationship
                for i in range(Batch_size):
                    pairs = len(e_posi[i][0]) * len(e_posi[i][1])  # for multiple words in entity
                    for e1 in e_posi[i][0]:
                        for e2 in e_posi[i][1]:
                            gt_posi = [e1, e2]
                            gt_posi.sort()
                            gt_result = gt_posi[0] * Relation_type + r_label[i]
                            #print(gt_posi)
                            #print(gt_result)
                            #input()
                            u = RE(ctx[i:i + 1, :gt_posi[1] + 1, :])
                            result = nn.Softmax(dim=-1)(u[0, :, :].view(-1))
                            #print(result)
                            for i, th in enumerate(Relation_threshold):
                                if result[gt_result].item() > th:
                                    TP[i][r_label[i]] += 1 / pairs
                                else:
                                    _, false_class = torch.max(u[0, gt_posi[0], :], dim=0)
                                    FN[i][r_label[i]] += 1 / pairs
                                    FP[i][false_class] += 1 / pairs

            print("[epoch: %d] \n", flush=True)

            for i, th in enumerate(Relation_threshold):
                for r in range(Relation_type):
                    F1[i][r] = (2 * TP[i][r] + epsilon) / (2 * TP[i][r] + FP[i][r] + FN[i][r] + epsilon)
                total_F1[i] = np.average(np.array(F1[i]))
                micro_F1[i] = (2 * sum(TP[i]) + epsilon) / (2 * sum(TP[i]) + sum(FP[i]) + sum(FN[i]) + epsilon)
                print('(threshold %.2f) val ave F1: %.4f, val micro F1: %.4f' % (th, total_F1[i], micro_F1[i]), flush=True)

            with open(LOG_FILE, 'a+') as LogDump:
                LogWriter = csv.writer(LogDump)
                LogWriter.writerows(F1)

        if (e + 1) % Log_every == 0:
            torch.save(LSTM_layer.state_dict(), SAVE_DIR + 'LSTM_woemb_' + str(e))
            torch.save(RE.state_dict(), SAVE_DIR + 'RE_woemb_' + str(e))



def train():
    training_ls = dir_reader(TRAIN_DIR)
    # test_ls = dir_reader(TEST_DIR)
    relations = relation_reader(cache=RELATIONS)
    assert relations['Other'] == 0
    assert Relation_type == len(relations)
    print(relations)

    train_ls, val_ls = training_ls[:len(training_ls) - 800], training_ls[len(training_ls) - 800:]
    #train_ls, val_ls = training_ls[:2], training_ls[:10]
    train_data = tokenizer((train_ls, TRAIN_DIR), relations, pretrain_type='elmo_repre')
    val_data = tokenizer((val_ls, TRAIN_DIR), relations, pretrain_type='elmo_repre')
    # test_data = tokenizer((test_ls, TEST_DIR), relations, pretrain_type='elmo_repre')

    print('%d training data, %d validation data' % (len(train_data.data), len(val_data.data)), flush=True)

    LSTM_layer = SeqLayer(ELMo_size, Hidden_size, Hidden_layer, Dropout, Bidirection).cuda()
    RE = RelationDetect_woemb(Hidden_size, Relation_type, Hidden_size, Dropout).cuda()

    print('network initialized', flush=True)

    #LSTM_layer.load_state_dict(torch.load(SAVE_DIR + 'LSTM_499'))
    #RE.load_state_dict(torch.load(SAVE_DIR + 'RE_499'))

    if os.path.isdir(LOG_FILE):
        os.rmdir(LOG_FILE)

    end2end(train_data, val_data, LSTM_layer, RE, Learning_rate, Epoch)

    '''
    NER_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    RE_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    NER_optimizer = optim.Adam(NER.parameters(), lr=Learning_rate, weight_decay=Weight_decay)
    RE_optimizer = optim.Adam(RE.parameters(), lr=Learning_rate, weight_decay=Weight_decay)    

    for e in range(Epoch):
        train_data.reset_epoch()
        NER.train()
        RE.train()
        NER_losses = []
        RE_losses = []
        while not train_data.epoch_finish:
            NER_optimizer.zero_grad()
            RE_optimizer.zero_grad()
            standard_emb, e_label, e_posi, r_label, seq_length, mask, seq_pos = train_data.get_batch(Batch_size)
            z, y, y_out, b = NER(standard_emb)

            entity_loss = 0
            relation_loss = 0

            for s in range(1, max(seq_length) + 1): # s is the count of word number
                u = RE(z[:, :s, :], b[:, :s, :])
                relation_teacher = torch.zeros((Batch_size, s), dtype=torch.long, requires_grad=False).cuda()
                for i in range(Batch_size):
                    if s > seq_length[i]:
                        relation_teacher[i, :] = -1
                    if s - 1 in e_posi[i][1] and e_posi[i][1][0] > e_posi[i][0][0]:
                        relation_teacher[i, e_posi[i][0]] = r_label[i]
                    elif s - 1 in e_posi[i][0] and e_posi[i][0][0] > e_posi[i][1][0]:
                        relation_teacher[i, e_posi[i][1]] = r_label[i]

                entity_loss += NER_criterion(y[:, :s, :].contiguous().view(s * Batch_size, -1), e_label[:, :s].contiguous().view(-1))
                relation_loss += RE_criterion(u.contiguous().view(s * Batch_size, -1), relation_teacher.contiguous().view(-1))

            total_loss = entity_loss + relation_loss
            total_loss.backward()
            NER_optimizer.step()
            RE_optimizer.step()
            NER_losses.append(entity_loss.item())
            RE_losses.append(relation_loss.item())

        if (e + 1) % Val_every == 0:
            val_data.reset_epoch()
            NER.eval()
            RE.eval()
            TP = [0.] * len(relations)
            FP = [0.] * len(relations)
            FN = [0.] * len(relations)
            F1 = [0.] * len(relations)
            while not val_data.epoch_finish:
                standard_emb, e_label, e_posi, r_label, seq_length, mask, seq_pos = val_data.get_batch(Batch_size)
                z, y, y_out, b = NER(standard_emb)

                for i in range(Batch_size):
                    pairs = len(e_posi[i][0]) * len(e_posi[i][1]) # for multiple words in entity
                    for e1 in e_posi[i][0]:
                        for e2 in e_posi[i][1]:
                            gt_posi = [e1, e2]
                            gt_posi.sort()
                            u = RE(z[i:i+1, :gt_posi[1], :], b[i:i+1, :gt_posi[1], :])
                            _, result = torch.max(u[0, gt_posi[0], :], dim=0)
                            if result.item() == r_label[i]:
                                TP[r_label[i]] += 1./pairs
                            else:
                                FN[r_label[i]] += 1./pairs
                                FP[result.item()] += 1./pairs

            for r in range(len(relations)):
                F1[r] = 2 * TP[r] / (2 * TP[r] + FP[r] + FN[r])

            with open(LOG_FILE, 'w+') as LogDump:
                LogWriter = csv.writer(LogDump)
                LogWriter.writerow(F1)
            total_F1 = np.average(np.array(F1))
            micro_F1 = 2 * sum(TP) / (2 * sum(TP) + sum(FP) + sum(FN))

            print('[epoch: %d] NER loss: %.4f, RE loss: %.4f, val ave F1: %.4f, val micro F1: %.4f' %
                  (e, np.average(np.array(NER_losses)), np.average(np.array(RE_losses)), total_F1, micro_F1), flush=True)

        if (e + 1) % Log_every == 0:
            torch.save(NER.state_dict(), SAVE_DIR + 'NER_' + str(e))
            torch.save(RE.state_dict(), SAVE_DIR + 'RE_' + str(e))
        '''



if __name__ == "__main__":
    train()
