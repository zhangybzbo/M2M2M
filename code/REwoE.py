import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from model import SeqLayer, RelationDetect_woemb
from utils import dir_reader, relation_reader, tokenizer

epsilon = sys.float_info.epsilon

SAVE_DIR = 'models/snapshots/'
LOG_FILE = 'models/val_wo_entity_loss.csv'
TEST_LOG_FILE = 'models/val_wo_entity_test_loss.csv'
TRAIN_DIR = 'corpus/train/'
TEST_DIR = 'corpus/test/'
RELATIONS = 'data/relations.txt'
MODEL_NAME = '_noNER_'

Relation_threshold = [0.1, 0.2, 0.3, 0.4, 0.5]
Relation_type = 10
Max_seq_len = 100
ELMo_size = 1024
Hidden_size = 100
Hidden_layer = 3
Dropout = 0.5
Bidirection = True

Learning_rate = 0.0005
Weight_decay = 0.0005
Epoch = 1000
Batch_size = 50
Val_every = 50
Log_every = 50


def get_REteacher(s, length, e_posi, relation):
    teacher = []
    count = []
    for i in range(Batch_size):
        if s - 1 in e_posi[i][1] and e_posi[i][1][0] > e_posi[i][0][0]:
            tmp = [p * Relation_type + relation[i].item() for p in e_posi[i][0]]
        elif s - 1 in e_posi[i][0] and e_posi[i][0][0] > e_posi[i][1][0]:
            tmp = [p * Relation_type + relation[i].item() for p in e_posi[i][1]]
        else:
            tmp = [-1]  # ignore NER
        '''elif s > length[i]:
            tmp = [-1]
        else:
            tmp = [(s - 1) * Relation_type]'''
        count.append(len(tmp))
        teacher.append(tmp)
    return teacher, count


def end2end(train_data, val_data, LSTM_layer, RE, lr, epoch):
    RE_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    LSTM_optimizer = optim.Adam(LSTM_layer.parameters(), lr=lr, weight_decay=Weight_decay)
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
                        relation_loss += RE_criterion(u[i, :, :].view(1, -1), teacher_ij.view(1)) / re_count[i] #* s
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
            TP = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
            FP = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
            FN = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
            F1 = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
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
                            for j, th in enumerate(Relation_threshold):
                                if result[gt_result].item() > th:
                                    TP[j][r_label[i]] += 1 / pairs
                                else:
                                    _, false_class = torch.max(u[0, gt_posi[0], :], dim=0)
                                    FN[j][r_label[i]] += 1 / pairs
                                    FP[j][false_class] += 1 / pairs

            print("[epoch: %d]" % e, flush=True)

            for j, th in enumerate(Relation_threshold):
                for r in range(Relation_type):
                    F1[j][r] = (2 * TP[j][r] + epsilon) / (2 * TP[j][r] + FP[j][r] + FN[j][r] + epsilon)
                total_F1[j] = np.average(np.array(F1[j]))
                micro_F1[j] = (2 * sum(TP[j]) + epsilon) / (2 * sum(TP[j]) + sum(FP[j]) + sum(FN[j]) + epsilon)
                print('(threshold %.2f) val ave F1: %.4f, val micro F1: %.4f' % (th, total_F1[j], micro_F1[j]), flush=True)

            with open(LOG_FILE, 'a+') as LogDump:
                LogWriter = csv.writer(LogDump)
                LogWriter.writerows(F1)

        if (e + 1) % Log_every == 0:
            torch.save(LSTM_layer.state_dict(), SAVE_DIR + 'LSTM' + MODEL_NAME + str(e))
            torch.save(RE.state_dict(), SAVE_DIR + 'RE' + MODEL_NAME + str(e))



def train():
    training_ls = dir_reader(TRAIN_DIR)
    relations = relation_reader(cache=RELATIONS)
    assert relations['Other'] == 0
    assert Relation_type == len(relations)
    print(relations)

    train_ls, val_ls = training_ls[:len(training_ls) - 800], training_ls[len(training_ls) - 800:]
    #train_ls, val_ls = training_ls[:2], training_ls[:10]
    train_data = tokenizer((train_ls, TRAIN_DIR), relations, pretrain_type='elmo_repre')
    val_data = tokenizer((val_ls, TRAIN_DIR), relations, pretrain_type='elmo_repre')

    print('%d training data, %d validation data' % (len(train_data.data), len(val_data.data)), flush=True)

    LSTM_layer = SeqLayer(ELMo_size, Hidden_size, Hidden_layer, Dropout, Bidirection).cuda()
    RE = RelationDetect_woemb(Hidden_size, Relation_type, Hidden_size, Dropout).cuda()

    print('network initialized', flush=True)

    #LSTM_layer.load_state_dict(torch.load(SAVE_DIR + 'LSTM_499'))
    #RE.load_state_dict(torch.load(SAVE_DIR + 'RE_499'))

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    end2end(train_data, val_data, LSTM_layer, RE, Learning_rate, Epoch)


def test():
    test_ls = dir_reader(TEST_DIR)
    relations = relation_reader(cache=RELATIONS)
    assert relations['Other'] == 0
    assert Relation_type == len(relations)
    print(relations)

    test_data = tokenizer((test_ls, TEST_DIR), relations, pretrain_type='elmo_repre')
    print('%d test data' % len(test_data.data), flush=True)

    LSTM_layer = SeqLayer(ELMo_size, Hidden_size, Hidden_layer, Dropout, Bidirection).cuda()
    RE = RelationDetect_woemb(Hidden_size, Relation_type, Hidden_size, Dropout).cuda()

    LSTM_layer.load_state_dict(torch.load(SAVE_DIR + 'LSTM' + MODEL_NAME + '999'))
    RE.load_state_dict(torch.load(SAVE_DIR + 'RE' + MODEL_NAME + '999'))

    print('network initialized', flush=True)

    if os.path.exists(TEST_LOG_FILE):
        os.remove(TEST_LOG_FILE)

    test_data.reset_epoch()
    LSTM_layer.eval()
    RE.eval()
    TP = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
    FP = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
    FN = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
    F1 = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
    Precision = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
    Recall = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
    total_F1 = [0.] * len(Relation_threshold)
    micro_F1 = [0.] * len(Relation_threshold)
    total_F1_9 = [0.] * len(Relation_threshold)
    micro_F1_9 = [0.] * len(Relation_threshold)
    macro_F1_9 = [0.] * len(Relation_threshold)
    precision_9 = [0.] * len(Relation_threshold)
    recall_9 = [0.] * len(Relation_threshold)
    while not test_data.epoch_finish:
        standard_emb, e_label, e_posi, r_label, seq_length, mask, seq_pos = test_data.get_batch(Batch_size)
        #print(standard_emb.size())
        #print(e_label)
        #print(e_posi, r_label, seq_length)
        #input()
        ctx = LSTM_layer(standard_emb, seq_length)

        # get relationship
        for i in range(Batch_size):
            '''# take NER into computation
            for s in range(1, seq_length[i] + 1):  # s is the count of word number
                if s - 1 in e_posi[i][0] and e_posi[i][0][0] > e_posi[i][1][0]:
                    gts = [(posi, r_label[i]) for posi in e_posi[i][1]]
                elif s - 1 in e_posi[i][1] and e_posi[i][1][0] > e_posi[i][0][0]:
                    gts = [(posi, r_label[i]) for posi in e_posi[i][0]]
                else:
                    gts = [((s - 1), 0)]
                #print(gts)

                u = RE(ctx[i:i + 1, :s, :])
                result = nn.Softmax(dim=-1)(u[0, :, :].view(-1))
                #print(result)
                #print(result.size())
                #input()

                for j, th in enumerate(Relation_threshold):
                    candidates = (result > th).nonzero()
                    #print(candidates)
                    for location, rtype in gts:
                        gt = location * Relation_type + rtype
                        if gt in candidates:
                            # correct entity correct relation
                            TP[j][rtype] += 1
                            candidates = candidates[candidates != gt]
                        else:
                            # at least one is wrong
                            FN[j][rtype] += 1
                    for candidate in candidates:
                        gt_locations = [l for (l, rt) in gts]
                        if candidate // Relation_type in gt_locations:
                            # correct entity wrong relation, omit
                            continue
                        else:
                            # wrong entity
                            FP[j][candidate % Relation_type] += 1
                    #print(TP[j])
                    #print(FN[j])
                    #print(FP[j])
                    #input()'''


            # ignore NER
            if e_posi[i][0][0] > e_posi[i][1][0]:
                s = e_posi[i][0][0]
                gts = [posi * Relation_type + r_label[i] for posi in e_posi[i][1]]
                gtp = [posi for posi in e_posi[i][1]]
            else:
                s = e_posi[i][1][0]
                gts = [posi * Relation_type + r_label[i] for posi in e_posi[i][0]]
                gtp = [posi for posi in e_posi[i][0]]

            u = RE(ctx[i:i + 1, :s + 1, :])
            result = nn.Softmax(dim=-1)(u[0, :, :].view(-1))

            for j, th in enumerate(Relation_threshold):
                candidates = (result > th).nonzero()
                # print(candidates)
                for candidate in candidates:
                    if candidate in gts:
                        # correct entity correct relation
                        TP[j][r_label[i]] += 1
                    else:
                        # at least one is wrong
                        FN[j][r_label[i]] += 1
                        FP[j][candidate % Relation_type] += 1



    for j, th in enumerate(Relation_threshold):
        for r in range(Relation_type):
            F1[j][r] = (2 * TP[j][r] + epsilon) / (2 * TP[j][r] + FP[j][r] + FN[j][r] + epsilon)
            Precision[j][r] = (TP[j][r] + epsilon) / (TP[j][r] + FP[j][r] + epsilon)
            Recall[j][r] = (TP[j][r] + epsilon) / (TP[j][r] + FN[j][r] + epsilon)
        total_F1[j] = np.average(np.array(F1[j]))
        micro_F1[j] = (2 * sum(TP[j]) + epsilon) / (2 * sum(TP[j]) + sum(FP[j]) + sum(FN[j]) + epsilon)
        total_F1_9[j] = np.average(np.array(F1[j][1:]))
        micro_F1_9[j] = (2 * sum(TP[j][1:]) + epsilon) / (2 * sum(TP[j][1:]) + sum(FP[j][1:]) + sum(FN[j][1:]) + epsilon)
        precision_9[j] = np.average(np.array(Precision[j][1:]))
        recall_9[j] = np.average(np.array(Recall[j][1:]))
        macro_F1_9[j] = (2 * recall_9[j] * precision_9[j] + epsilon) / (recall_9[j] + precision_9[j] + epsilon)
        print('(threshold %.2f)' % th, flush=True)
        print('with other: ave F1: %.4f, micro F1: %.4f' % (total_F1[j], micro_F1[j]), flush=True)
        print('without other: ave F1: %.4f, micro F1: %.4f, macro F1: %.4f, ave precision: %.4f, ave recall: %.4f'
              % (total_F1_9[j], micro_F1_9[j], macro_F1_9[j], precision_9[j], recall_9[j]), flush=True)

    with open(TEST_LOG_FILE, 'a+') as LogDump:
        LogWriter = csv.writer(LogDump)
        LogWriter.writerows(F1)



if __name__ == "__main__":
    # train()
    test()
