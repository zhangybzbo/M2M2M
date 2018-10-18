import numpy as np
import re
import torch

file_list = ['test_0.csv', 'test_1.csv', 'test_2.csv', 'test_3.csv', 'test_4.csv',
             'train_0.csv', 'train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv']
file_dir = 'data/'
save_code = 'codels.txt'
save_word = 'wordls.txt'


def read_file():
    # get code & word list
    # dump into two files
    fw = open(file_dir + save_code, "w")
    fw_2 = open(file_dir + save_word, 'w')
    code_list = []
    word_list = []
    for f in file_list:
        with open(file_dir + f) as fr:
            for line in fr.readlines():
                code = line.strip().split('\t')[1]
                phrase = line.strip().split('\t')[0]
                words = re.split(' |,|\)|\(|-|/|\.|\'|\"', phrase.strip())
                if code not in code_list:
                    code_list.append(code)
                    fw.write(code + '\n')
                for word in words:
                    if (word.strip() not in word_list) and (not word.strip() == ''):
                        word_list.append(word.strip())
                        fw_2.write(word + '\n')

    fw.close()
    fw_2.close()
    print(len(code_list))
    print(len(word_list))


def word_to_index(vocab):
    # word list to index
    # {'word': index}
    list_to_index = {}
    for i, word in enumerate(vocab):
        list_to_index[word] = i + 1
        # for word list, 0=padding

    return list_to_index


def code_to_index(vocab):
    # word list to index
    # {'word': index}
    list_to_index = {}
    for i, word in enumerate(vocab):
        list_to_index[word] = i

    return list_to_index


def read_vocab(path):
    # read list from file
    # [n * 'word']
    vocab = []
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


class tokenizer(object):
    # token data 
    # data: [n * {'phrase': [word_id_list], 'code': code_id}
    def __init__(self, wordls, codels, datafile):
        self.data = []
        with open(datafile) as f:
            for line in f.readlines():
                new_data = dict()
                wordtok = []
                code = line.strip().split('\t')[1]
                phrase = line.strip().split('\t')[0]
                words = re.split(' |,|\)|\(|-|/|\.|\'|\"', phrase.strip())
                for word in words:
                    if not word.strip() == '':
                        wordtok.append(wordls[word.strip()])
                new_data['phrase'] = wordtok
                new_data['code'] = codels[code]
                self.data.append(new_data)

        self.epoch_finish = False
        self.position = 0

    def reset_epoch(self):
        self.epoch_finish = False
        self.position = 0

    def get_batch(self, batch_size):
        batch = self.data[self.position:self.position + batch_size]
        seq = []
        seq_length = [len(element['phrase']) for element in batch]
        mask = []
        for element in batch:
            encode = element['phrase']
            l = len(encode)
            encode += [0] * (max(seq_length) - l)
            seq.append(encode)
            mask.append([0] * l + [1] * (max(seq_length) - l))

        label = [element['code'] for element in batch]

        seq = torch.tensor(seq, requires_grad=False).cuda()
        label = torch.tensor(label, requires_grad=False).cuda()
        mask = torch.tensor(mask, requires_grad=False).byte().cuda()

        self.position += batch_size
        if (self.position + batch_size) > len(self.data):
            self.epoch_finish = True

        return seq, label, seq_length, mask


def read_embed(path):
    # read embedding from file
    # {'word': [embedding]}
    embed = {}
    max_e = 0
    min_e = 0
    with open(path) as f:
        f.readline()
        for line in f.readlines():
            ls = line.strip().split()
            embed[ls[0].strip()] = [float(i.strip()) for i in ls[1:]]
            max_t = max(embed[ls[0].strip()])
            min_t = min(embed[ls[0].strip()])
            if max_t > max_e:
                max_e = max_t
            if min_t < min_e:
                min_e = min_t

    return embed, max_e, min_e


def pre_embed(raw_embedding, word_vocab, max_e, min_e, embedding_size):
    # permute embedding to word id
    # [n * embedding_size]
    embedding = []
    embedding.append([0.] * embedding_size)
    for i, word in enumerate(word_vocab):
        if word in raw_embedding.keys():
            embedding.append(raw_embedding[word])
        else:
            # print("%s not in embedding" % word)
            rand = list(np.random.uniform(min_e, max_e, embedding_size))
            embedding.append(rand)

    return embedding


if __name__ == "__main__":
    read_file()
