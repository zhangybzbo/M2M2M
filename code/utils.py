import os
import random
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
# from nltk.tokenize.stanford import StanfordTokenizer

random.seed(1)

elmo_options = 'models/elmo_2x4096_512_2048cnn_2xhighway_options.json'
elmo_weights = 'models/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

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


def word_reader(filels=None, cache=None):
    word2index = {}

    if filels:
        i = 0
        with open(cache) as f:
            for i, w in enumerate(f.readlines()):
                word2index[w.strip()] = i
        fw = open(cache, "a+")
        for id in filels[0]:
            with open(filels[1] + id + '.txt') as ftxt:
                line = ftxt.readlines()
                assert len(line) == 1
                line = line[0].strip()
                words = line.split(' ')
                for word in words:
                    if (word.strip().lower() not in word2index) and (not word.strip() == ''):
                        word2index[word.strip().lower()] = i
                        i += 1
                        fw.write(word.strip().lower() + '\n')
        fw.close()

    elif cache:
        with open(cache) as fw:
            for i, w in enumerate(fw.readlines()):
                word2index[w.strip()] = i

    return word2index


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
                        new_data['labels'][i] = 1
                        new_data['entity_posi'][1].append(i)
                    posi += len(new_data['sentence'][i]) + 1
                    if posi > t1_l and posi > t2_l:
                        break
                # assert 1 in new_data['labels'] and 2 in new_data['labels'], (id, new_data['labels'], line, lines)

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


if __name__ == '__main__':
    ls = dir_reader('corpus/train/')
    word2index = word_reader((ls, 'corpus/train/'), 'data/corpus.txt')
    print(word2index)
    print(len(word2index))
    input()
    ls = dir_reader('corpus/test/')
    word2index = word_reader((ls, 'corpus/test/'), 'data/corpus.txt')
    print(word2index)
    print(len(word2index))