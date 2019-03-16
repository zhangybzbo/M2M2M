import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from model import SeqLayer, EntityDetect, RelationDetect
from utils import dir_reader, relation_reader, tokenizer

epsilon = sys.float_info.epsilon

SAVE_DIR = 'models/snapshots/'
LOG_FILE = 'models/val.csv'
TEST_LOG_FILE = 'models/val_test.csv'
TRAIN_DIR = 'corpus/train/'
TEST_DIR = 'corpus/test/'
RELATIONS = 'data/relations.txt'

Relation_threshold = [0.3, 0.4, 0.5]
Relation_type = 10
Max_seq_len = 100
ELMo_size = 1024
Label_embed = 25
Hidden_size = 100
Hidden_layer = 3
Dropout = 0.5
Bidirection = True

Learning_rate = 0.0001
LR_decay = 10
Weight_decay = 0.0005
Epoch = 1000
Batch_size = 50
Val_every = 20
Log_every = 20


def pretrain_NER(train_data, val_data, LSTM_layer, NER, lr, epoch):
    NER_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    LSTM_optimizer = optim.Adam(LSTM_layer.parameters(), lr=lr, weight_decay=Weight_decay)
    NER_optimizer = optim.Adam(NER.parameters(), lr=lr, weight_decay=Weight_decay)

    for e in range(epoch):
        train_data.reset_epoch()
        LSTM_layer.train()
        NER.train()
        while not train_data.epoch_finish:
            LSTM_optimizer.zero_grad()
            NER_optimizer.zero_grad()
            standard_emb, e_label, e_posi, _, seq_length, _, _ = train_data.get_batch(Batch_size)

            ctx = LSTM_layer(standard_emb, seq_length)

            y_gold = torch.zeros(Batch_size, requires_grad=False).long().cuda()
            entity_loss = 0.

            for s in range(max(seq_length)):
                tmp_ctx = [ctx[i, s, :] for i in range(Batch_size) if s < seq_length[i]]
                tmp_ctx = torch.stack(tmp_ctx)

                _, logit, y_out = NER(tmp_ctx, y_gold)

                y_gt = [e_label[i, s] for i in range(Batch_size) if s < seq_length[i]]
                y_gt = torch.stack(y_gt)
                entity_loss += NER_criterion(logit, y_gt)
                if s + 1 < max(seq_length):
                    y_gold = [e_label[i, s] for i in range(Batch_size) if s + 1 < seq_length[i]]
                    y_gold = torch.stack(y_gold)

            entity_loss.backward()
            LSTM_optimizer.step()
            NER_optimizer.step()

        if (e + 1) % Val_every == 0:
            val_data.reset_epoch()
            LSTM_layer.eval()
            NER.eval()
            count_all = 0
            correct_raw = 0
            correct_acc = 0
            while not val_data.epoch_finish:
                standard_emb, e_label, _, _, seq_length, _, _ = val_data.get_batch(Batch_size)
                ctx = LSTM_layer(standard_emb, seq_length)
                y_out = torch.zeros(Batch_size, requires_grad=False).long().cuda()
                y_all = torch.zeros((Batch_size, max(seq_length)), requires_grad=False).long().cuda()
                for s in range(max(seq_length)):
                    _, logit, y_out = NER(ctx[:, s, :], y_out)
                    for i in range(Batch_size):
                        y_all[i, s] = y_out[i].detach() if s < seq_length[i] else -1

                for i in range(Batch_size):
                    count_all += 1
                    correct_acc += torch.all(
                        torch.eq(y_all[i, :seq_length[i]], e_label[i, :seq_length[i]])).long().item()
                    e1 = y_all[i, :seq_length[i]].nonzero()
                    e2 = e_label[i, :seq_length[i]].nonzero()
                    correct_raw += int(torch.equal(e1, e2))

            print("[epoch: %d] entity detection accuracy: raw %.4f, accurate %.4f" %
                  (e, correct_raw / float(count_all), correct_acc / float(count_all)), flush=True)

        if (e + 1) % Log_every == 0:
            torch.save(LSTM_layer.state_dict(), SAVE_DIR + 'LSTM_pretrain_2_' + str(e+1000))
            torch.save(NER.state_dict(), SAVE_DIR + 'NER_pretrain_2_' + str(e+1000))


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


def end2end(train_data, val_data, LSTM_layer, NER, RE, lr, epoch):
    NER_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    RE_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    LSTM_optimizer = optim.Adam(LSTM_layer.parameters(), lr=lr/LR_decay, weight_decay=Weight_decay)
    NER_optimizer = optim.Adam(NER.parameters(), lr=lr/LR_decay, weight_decay=Weight_decay)
    RE_optimizer = optim.Adam(RE.parameters(), lr=lr, weight_decay=Weight_decay)

    for e in range(epoch):
        train_data.reset_epoch()
        LSTM_layer.train()
        NER.train()
        RE.train()
        NER_losses = []
        RE_losses = []
        while not train_data.epoch_finish:
            LSTM_optimizer.zero_grad()
            NER_optimizer.zero_grad()
            RE_optimizer.zero_grad()
            standard_emb, e_label, e_posi, r_label, seq_length, mask, seq_pos = train_data.get_batch(Batch_size)
            #print(standard_emb.size())
            #print(e_label)
            #print(e_posi, r_label, seq_length)
            #input()

            ctx = LSTM_layer(standard_emb, seq_length)
            #print(ctx.size())
            #input()
            label_emb = torch.zeros((Batch_size, max(seq_length), Label_embed), requires_grad=False).cuda()

            y_gold = torch.zeros(Batch_size, requires_grad=False).long().cuda() # gold label or whatever label,from last time step
            entity_loss = 0.

            # get entity detection and label embedding
            for s in range(max(seq_length)):
                v_tp, logit, y_out = NER(ctx[:, s, :], y_gold)
                entity_loss += NER_criterion(logit, e_label[:, s]) * (max(seq_length) - s)

                y_gold = torch.zeros(Batch_size, requires_grad=False).long().cuda()
                for i in range(Batch_size):
                    if s < seq_length[i]:
                        y_gold[i] = e_label[i, s]
                    if s > 0 and s <= seq_length[i]:
                        label_emb[i, s - 1, :] = v_tp[i, :] # record embedding of label of last time step

            # get label embedding of the last step
            v_tp, _, _ = NER(torch.zeros(Batch_size, Hidden_size).cuda(), y_gold)
            for i in range(Batch_size):
                if seq_length[i] == max(seq_length):
                    label_emb[i, -1, :] = v_tp[i, :]

            #print(label_emb[:,:,0])
            #input()

            relation_loss = 0.
            # relationship
            for s in range(1, max(seq_length) + 1):  # s is the count of word number
                u = RE(ctx[:, :s, :], label_emb[:, :s, :])
                #print(u.size())
                re_teacher, re_count = get_REteacher(s, seq_length, e_posi, r_label)
                #print(re_teacher, re_count)
                for i in range(Batch_size):
                    for j in range(re_count[i]):
                        teacher_ij = torch.tensor(re_teacher[i][j], dtype=torch.long, requires_grad=False).cuda()
                        relation_loss += RE_criterion(u[i, :, :].view(1, -1), teacher_ij.view(1)) / re_count[i]
                        #print(i,j,teacher_ij,relation_loss)
                        #input()

            total_loss = entity_loss + relation_loss
            total_loss.backward()
            LSTM_optimizer.step()
            NER_optimizer.step()
            RE_optimizer.step()
            NER_losses.append(entity_loss.item())
            RE_losses.append(relation_loss.item())


        if (e + 1) % Val_every == 0:
            val_data.reset_epoch()
            LSTM_layer.eval()
            NER.eval()
            RE.eval()
            TP = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
            FP = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
            FN = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
            F1 = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
            total_F1 = [0.] * len(Relation_threshold)
            micro_F1 = [0.] * len(Relation_threshold)
            count_all = 0
            correct_raw = 0
            correct_acc = 0
            while not val_data.epoch_finish:
                standard_emb, e_label, e_posi, r_label, seq_length, mask, seq_pos = val_data.get_batch(Batch_size)
                #print(standard_emb.size())
                #print(e_label)
                #print(e_posi, r_label, seq_length)
                #input()
                ctx = LSTM_layer(standard_emb, seq_length)

                label_emb = torch.zeros((Batch_size, max(seq_length), Label_embed), requires_grad=False).cuda()
                y_out = torch.zeros(Batch_size, requires_grad=False).long().cuda()
                y_all = torch.zeros((Batch_size, max(seq_length)), requires_grad=False).long().cuda()
                for s in range(max(seq_length)):
                    v_tp, logit, y_out = NER(ctx[:, s, :], y_out)
                    for i in range(Batch_size):
                        y_all[i, s] = y_out[i].detach() if s < seq_length[i] else -1
                        if s > 0 and s <= seq_length[i]:
                            label_emb[i, s - 1, :] = v_tp[i, :].detach()  # record embedding of label of last time step

                # get label embedding of the last step
                v_tp, _, _ = NER(torch.zeros(Batch_size, Hidden_size).cuda(), y_out)
                for i in range(Batch_size):
                    if seq_length[i] == max(seq_length):
                        label_emb[i, -1, :] = v_tp[i, :].detach()

                #print(y_all)
                #print(e_label)
                #print(label_emb[:, :, 0])
                #input()

                # compute entity detection accuracy
                for i in range(Batch_size):
                    count_all += 1
                    correct_acc += torch.all(
                        torch.eq(y_all[i, :seq_length[i]], e_label[i, :seq_length[i]])).long().item()
                    e1 = y_all[i, :seq_length[i]].nonzero()
                    e2 = e_label[i, :seq_length[i]].nonzero()
                    correct_raw += int(torch.equal(e1, e2))

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
                            u = RE(ctx[i:i + 1, :gt_posi[1] + 1, :], label_emb[i:i + 1, :gt_posi[1] + 1, :])
                            result = nn.Softmax(dim=-1)(u[0, :, :].view(-1))
                            #print(result)
                            for j, th in enumerate(Relation_threshold):
                                if result[gt_result].item() > th:
                                    TP[j][r_label[i]] += 1 / pairs
                                else:
                                    _, false_class = torch.max(u[0, gt_posi[0], :], dim=0)
                                    FN[j][r_label[i]] += 1 / pairs
                                    FP[j][false_class] += 1 / pairs

            print("[epoch: %d] \nentity detection accuracy: raw %.4f, accurate %.4f" %
                  (e, correct_raw / count_all, correct_acc / count_all), flush=True)
            print('NER loss: %.4f, RE loss: %.4f' % (np.average(np.array(NER_losses)), np.average(np.array(RE_losses))),
                  flush=True)

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
            torch.save(LSTM_layer.state_dict(), SAVE_DIR + 'LSTM_2_' + str(e))
            torch.save(NER.state_dict(), SAVE_DIR + 'NER_2_' + str(e))
            torch.save(RE.state_dict(), SAVE_DIR + 'RE_2_' + str(e))



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
    NER = EntityDetect(Label_embed, Hidden_size, 3, Dropout).cuda()
    RE = RelationDetect(Hidden_size, Label_embed, Relation_type, Hidden_size, Dropout).cuda()

    print('network initialized', flush=True)

    LSTM_layer.load_state_dict(torch.load(SAVE_DIR + 'LSTM_pretrain_2_1339'))
    NER.load_state_dict(torch.load(SAVE_DIR + 'NER_pretrain_2_1339'))
    # RE.load_state_dict(torch.load(SAVE_DIR + 'RE_499'))

    # pretrain_NER(train_data, val_data, LSTM_layer, NER, Learning_rate, Epoch)

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    end2end(train_data, val_data, LSTM_layer, NER, RE, Learning_rate, Epoch)

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

def test():
    test_ls = dir_reader(TEST_DIR)
    relations = relation_reader(cache=RELATIONS)
    assert relations['Other'] == 0
    assert Relation_type == len(relations)
    print(relations)

    test_data = tokenizer((test_ls, TEST_DIR), relations, pretrain_type='elmo_repre')
    print('%d test data' % len(test_data.data), flush=True)

    LSTM_layer = SeqLayer(ELMo_size, Hidden_size, Hidden_layer, Dropout, Bidirection).cuda()
    NER = EntityDetect(Label_embed, Hidden_size, 3, Dropout).cuda()
    RE = RelationDetect(Hidden_size, Label_embed, Relation_type, Hidden_size, Dropout).cuda()

    LSTM_layer.load_state_dict(torch.load(SAVE_DIR + 'LSTM_2_999'))
    NER.load_state_dict(torch.load(SAVE_DIR + 'NER_2_999'))
    RE.load_state_dict(torch.load(SAVE_DIR + 'RE_2_999'))

    print('network initialized', flush=True)

    if os.path.exists(TEST_LOG_FILE):
        os.remove(TEST_LOG_FILE)

    test_data.reset_epoch()
    LSTM_layer.eval()
    NER.eval()
    RE.eval()
    TP = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
    FP = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
    FN = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
    F1 = [[0.] * Relation_type for _ in range(len(Relation_threshold))]
    total_F1 = [0.] * len(Relation_threshold)
    micro_F1 = [0.] * len(Relation_threshold)
    total_F1_9 = [0.] * len(Relation_threshold)
    micro_F1_9 = [0.] * len(Relation_threshold)
    precision_9 = [0.] * len(Relation_threshold)
    recall_9 = [0.] * len(Relation_threshold)
    count_all = 0
    correct_raw = 0

    while not test_data.epoch_finish:
        standard_emb, e_label, e_posi, r_label, seq_length, mask, seq_pos = test_data.get_batch(Batch_size)
        # print(standard_emb.size())
        # print(e_label)
        # print(e_posi, r_label, seq_length)
        # input()
        ctx = LSTM_layer(standard_emb, seq_length)

        label_emb = torch.zeros((Batch_size, max(seq_length), Label_embed), requires_grad=False).cuda()
        y_out = torch.zeros(Batch_size, requires_grad=False).long().cuda()
        y_all = torch.zeros((Batch_size, max(seq_length)), requires_grad=False).long().cuda()
        for s in range(max(seq_length)):
            v_tp, logit, y_out = NER(ctx[:, s, :], y_out)
            for i in range(Batch_size):
                y_all[i, s] = y_out[i].detach() if s < seq_length[i] else -1
                if s > 0 and s <= seq_length[i]:
                    label_emb[i, s - 1, :] = v_tp[i, :].detach()  # record embedding of label of last time step

        # get label embedding of the last step
        v_tp, _, _ = NER(torch.zeros(Batch_size, Hidden_size).cuda(), y_out)
        for i in range(Batch_size):
            if seq_length[i] == max(seq_length):
                label_emb[i, -1, :] = v_tp[i, :].detach()

        # print(y_all)
        # print(e_label)
        # print(label_emb[:, :, 0])
        # input()

        # compute entity detection accuracy
        for i in range(Batch_size):
            count_all += 1
            e1 = y_all[i, :seq_length[i]].nonzero()
            e2 = e_label[i, :seq_length[i]].nonzero()
            correct_raw += int(torch.equal(e1, e2))

        # get relationship
        for i in range(Batch_size):
            for s in range(1, seq_length[i] + 1):  # s is the count of word number
                if s - 1 in e_posi[i][0] and e_posi[i][0][0] > e_posi[i][1][0]:
                    gts = [(posi, r_label[i]) for posi in e_posi[i][1]]
                elif s - 1 in e_posi[i][1] and e_posi[i][1][0] > e_posi[i][0][0]:
                    gts = [(posi, r_label[i]) for posi in e_posi[i][0]]
                else:
                    gts = [((s - 1), 0)]

                u = RE(ctx[i:i + 1, :s, :], label_emb[i:i + 1, :s, :])
                result = nn.Softmax(dim=-1)(u[0, :, :].view(-1))

                for j, th in enumerate(Relation_threshold):
                    candidates = (result > th).nonzero()
                    for location, rtype in gts:
                        gt = location * Relation_type + rtype
                        if gt in candidates:
                            # correct entity correct relation
                            TP[j][rtype] += 1
                            candidates = candidates[candidates != gt]
                        elif gt not in candidates:
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

    for j, th in enumerate(Relation_threshold):
        for r in range(Relation_type):
            F1[j][r] = (2 * TP[j][r] + epsilon) / (2 * TP[j][r] + FP[j][r] + FN[j][r] + epsilon)
        total_F1[j] = np.average(np.array(F1[j]))
        micro_F1[j] = (2 * sum(TP[j]) + epsilon) / (2 * sum(TP[j]) + sum(FP[j]) + sum(FN[j]) + epsilon)
        total_F1_9[j] = np.average(np.array(F1[j][1:]))
        micro_F1_9[j] = (2 * sum(TP[j][1:]) + epsilon) / (2 * sum(TP[j][1:]) + sum(FP[j][1:]) + sum(FN[j][1:]) + epsilon)
        precision_9[j] = (sum(TP[j][1:]) + epsilon) / (sum(TP[j][1:]) + sum(FP[j][1:]) + epsilon)
        recall_9[j] = (sum(TP[j][1:]) + epsilon) / (sum(TP[j][1:]) + sum(FN[j][1:]) + epsilon)
        print('(threshold %.2f)' % th, flush=True)
        print('with other: val ave F1: %.4f, val micro F1: %.4f' % (total_F1[j], micro_F1[j]), flush=True)
        print('without other: val ave F1: %.4f, val micro F1: %.4f, precision: %.4f, recall: %.4f'
              % (total_F1_9[j], micro_F1_9[j], precision_9[j], recall_9[j]), flush=True)

    with open(TEST_LOG_FILE, 'a+') as LogDump:
        LogWriter = csv.writer(LogDump)
        LogWriter.writerows(F1)

if __name__ == "__main__":
    #train()
    test()
