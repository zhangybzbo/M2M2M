import os
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
# from nltk.tokenize.stanford import StanfordTokenizer
import random
from model import EntityDetect, RelationDetect

random.seed(1)

SAVE_DIR = 'models/'
TRAIN_DIR = 'corpus/train/'
TEST_DIR = 'corpus/test/'
RELATIONS = 'data/relations.txt'
elmo_options = 'models/elmo_2x4096_512_2048cnn_2xhighway_options.json'
elmo_weights = 'models/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

Max_seq_len = 100
ELMo_size = 1024
Label_embed = 100
Hidden_size = 100
Hidden_layer = 3
Dropout = 0.5

Learning_rate = 0.0001
Weight_decay = 0.0005
Epoch = 100
Batch_size = 20
Val_every = 5
Log_every = 5


# tokenizer = StanfordTokenizer(r'/playpen/home/zhangyb/M2M2M/LSTM-ER/data/common/stanford-postagger-2015-04-20/stanford-postagger.jar')

def dir_reader(file):
    ls = [id.split('.')[0] for id in os.listdir(file)]
    ls = list(set(ls))
    ls.sort(key=lambda id: int(id))
    return ls


def relation_reader(filels=None, cache=None):
    relation2index = {}

    if filels:
        i = 0
        fw = open(cache, "w")
        for id in filels[0]:
            with open(filels[1] + id + '.ann') as fann:
                lines = fann.readlines()
                assert len(lines) == 3
                relation_info = lines[2].strip().split('\t')[1].split(' ')
                assert len(relation_info) == 3
                if relation_info[0] not in relation2index:
                    relation2index[relation_info[0]] = i
                    i += 1
                    fw.write(relation_info[0] + '\n')

    elif cache:
        fw = open(cache)
        for i, re in enumerate(fw.readlines()):
            relation2index[re.strip()] = i

    return relation2index


class tokenizer(object):
    def __init__(self, filels, relationsls, pretrain_type=None):
        self.data = []
        self.pretrain = pretrain_type
        if pretrain_type == 'elmo_repre':
            self.pre_model = Elmo(elmo_options, elmo_weights, 2, dropout=0).cuda()
        elif pretrain_type == 'elmo_layer':
            self.pre_model = ElmoEmbedder(elmo_options, elmo_weights, cuda_device=0)

        for id in filels[0]:
            new_data = dict()

            with open(filels[1] + id + '.txt') as ftxt:
                line = ftxt.readlines()
                # print(line)
                assert len(line) == 1
                line = line[0].strip()
                words = line.split(' ')

                new_data['sentence'] = words
                new_data['length'] = len(words)
                new_data['position'] = [i + 1 for i in range(len(words))]

                if pretrain_type == 'elmo_repre':
                    elmo_id = batch_to_ids([words]).cuda()
                    pre_embed = self.pre_model(elmo_id)
                    new_data['emb'] = pre_embed['elmo_representations'][1].squeeze(0).detach()
                    # print(new_data['emb'].size())
                elif pretrain_type == 'elmo_layer':
                    pre_embed = self.pre_model.embed_sentence(words)
                    new_data['emb'] = torch.zeros((3, len(words), 1024), requires_grad=False)
                    for i in range(3):
                        new_data['emb'][i, :, :] = torch.from_numpy(pre_embed[i])

            with open(filels[1] + id + '.ann') as fann:
                lines = fann.readlines()
                # print(lines)
                assert len(lines) == 3
                relation_info = lines[2].strip().split('\t')[1].split(' ')
                assert len(relation_info) == 3
                if relation_info[1] == 'Arg1:T1' and relation_info[2] == 'Arg2:T2':
                    t1 = lines[0].strip().split('\t')
                    t2 = lines[1].strip().split('\t')
                elif relation_info[1] == 'Arg1:T2' and relation_info[2] == 'Arg2:T1':
                    t1 = lines[1].strip().split('\t')
                    t2 = lines[0].strip().split('\t')
                else:
                    assert False
                new_data['entities'] = [t1[2], t2[2]]
                new_data['entity_posi'] = [[], []]
                new_data['predicate'] = relationsls[relation_info[0]]

                # label sequence, 0 for others, 1 for t1, 2 for t2
                posi = 0
                new_data['labels'] = [0] * new_data['length']
                t1_b, t1_l = int(t1[1].split(' ')[1]), int(t1[1].split(' ')[2])
                t2_b, t2_l = int(t2[1].split(' ')[1]), int(t2[1].split(' ')[2])
                for i in range(new_data['length']):
                    if posi >= t1_b and posi + len(new_data['sentence'][i]) <= t1_l:
                        new_data['labels'][i] = 1
                        new_data['entity_posi'][0].append(i)
                    elif posi >= t2_b and posi + len(new_data['sentence'][i]) <= t2_l:
                        new_data['labels'][i] = 2
                        new_data['entity_posi'][1].append(i)
                    posi += len(new_data['sentence'][i]) + 1
                    if posi > t1_l and posi > t2_l:
                        break
                assert 1 in new_data['labels'] and 2 in new_data['labels'], (id, new_data['labels'], line, lines)

            self.data.append(new_data)
            # print(new_data)
            # input()

        self.max_length = max([l['length'] for l in self.data])
        self.epoch_finish = False
        self.position = 0

    def reset_epoch(self):
        self.epoch_finish = False
        self.position = 0
        random.shuffle(self.data)

    def get_batch(self, batch_size):
        batch = self.data[self.position:self.position + batch_size]
        seq_length = [element['length'] for element in batch]
        mask = []
        posi = []
        entity_label = []
        entity_posi = [element['entity_posi'] for element in batch]
        relation_label = [element['predicate'] for element in batch]

        if self.pretrain == 'elmo_repre':
            pre_model = torch.zeros((batch_size, max(seq_length), 1024), requires_grad=False).cuda()
        elif self.pretrain == 'elmo_layer':
            pre_model = torch.zeros((batch_size, 3, max(seq_length), 1024), requires_grad=False).cuda()
        else:
            pre_model = None

        for i, element in enumerate(batch):
            if self.pretrain == 'elmo_repre':
                pre_model[i, :element['length'], :] = element['emb']
            elif self.pretrain:
                pre_model[i, :, :element['length'], :] = element['emb']
            posi.append(element['position'] + [0] * (max(seq_length) - element['length']))
            mask.append([0] * element['length'] + [1] * (max(seq_length) - element['length']))
            entity_label.append(element['labels'] + [-1] * (max(seq_length) - element['length']))

        mask = torch.tensor(mask, requires_grad=False).byte().cuda()
        posi = torch.tensor(posi, requires_grad=False).cuda()
        entity_label = torch.tensor(entity_label, requires_grad=False).cuda()
        relation_label = torch.tensor(relation_label, requires_grad=False).cuda()

        self.position += batch_size
        if (self.position + batch_size) > len(self.data):
            self.epoch_finish = True

        return pre_model, entity_label, entity_posi, relation_label, seq_length, mask, posi


def train():
    training_ls = dir_reader(TRAIN_DIR)
    # test_ls = dir_reader(TEST_DIR)
    relations = relation_reader(cache=RELATIONS)
    assert relations['Other'] == 0
    print(relations)

    train_ls, val_ls = training_ls[:len(training_ls) - 800], training_ls[len(training_ls) - 800:]
    train_data = tokenizer((train_ls, TRAIN_DIR), relations, pretrain_type='elmo_repre')
    val_data = tokenizer((val_ls, TRAIN_DIR), relations, pretrain_type='elmo_repre')
    # test_data = tokenizer((test_ls, TEST_DIR), relations, pretrain_type='elmo_repre')

    print('%d training data, %d validation data' % (len(train_data.data), len(val_data.data)), flush=True)

    NER = EntityDetect(None, ELMo_size, Label_embed, Hidden_size, 3, Hidden_layer, Dropout).cuda()
    RE = RelationDetect(Hidden_size, Label_embed, len(relations), Hidden_size, Dropout).cuda()

    NER_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    RE_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    NER_optimizer = optim.Adam(NER.parameters(), lr=Learning_rate, weight_decay=Weight_decay)
    RE_optimizer = optim.Adam(RE.parameters(), lr=Learning_rate, weight_decay=Weight_decay)

    print('network initialized')

    LogDump = open(SAVE_DIR + 'val.csv', 'w')
    LogWriter = csv.writer(LogDump)

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
            print(e_label, e_posi, r_label)
            print(seq_length, mask, seq_pos)
            input()
            z, y, y_out, b = NER(standard_emb)

            entity_loss = 0
            relation_loss = 0

            for s in range(1, max(seq_length) + 1): # s is the count of word number
                u = RE(z[:, :s, :], b[:, :s, :])
                relation_teacher = torch.zeros((Batch_size, s), dtype=torch.long, requires_grad=False).cuda()
                for i in range(Batch_size):
                    if s > seq_length[i]:
                        relation_teacher[i, seq_length[i]:] = -1
                    if s - 1 in e_posi[i][1] and e_posi[i][1][0] > e_posi[i][0][0]:
                        relation_teacher[i, e_posi[i][0]] = r_label[i]
                    elif s - 1 in e_posi[i][0] and e_posi[i][0][0] > e_posi[i][1][0]:
                        relation_teacher[i, e_posi[i][1]] = r_label[i]

                entity_loss += NER_criterion(y[:, :s, :].contiguous().view(s * Batch_size, -1), e_label[:, :s].contiguous().view(-1))
                relation_loss += RE_criterion(u.contiguous().view(s * Batch_size, -1), relation_teacher.contiguous().view(-1))

            total_loss = entity_loss + relation_loss
            total_loss.backward()
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
            LogWriter.writerow(F1)
            total_F1 = np.average(np.array(F1))
            micro_F1 = 2 * sum(TP) / (2 * sum(TP) + sum(FP) + sum(FN))

            print('[epoch: %d] NER loss: %.4f, RE loss: %.4f, val ave F1: %.4f, val micro F1: %.4f' %
                  (e, np.average(np.array(NER_losses)), np.average(np.array(RE_losses)), total_F1, micro_F1), flush=True)

        if (e + 1) % Log_every == 0:
            torch.save(NER.state_dict(), SAVE_DIR + 'NER_' + str(e))
            torch.save(RE.state_dict(), SAVE_DIR + 'RE_' + str(e))

    LogDump.close()


if __name__ == "__main__":
    train()
