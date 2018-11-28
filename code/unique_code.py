import numpy as np
import re
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
import collections

file_list = ['test_0.csv', 'test_1.csv', 'test_2.csv', 'test_3.csv', 'test_4.csv',
             'train_0.csv', 'train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv']
file_dir = 'data/'
save_code = 'codels.txt'
save_word = 'wordls.txt'
elmo_options = 'models/elmo_2x4096_512_2048cnn_2xhighway_options.json'
elmo_weights = 'models/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'


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
                new_data['position'] = [i + 1 for i in range(len(wordtok))]
                new_data['code'] = codels[code]
                self.data.append(new_data)

        self.max_length = max([len(l['phrase']) for l in self.data])

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
        posi = []
        for element in batch:
            encode = element['phrase']
            l = len(encode)
            encode += [0] * (max(seq_length) - l)
            seq_position = element['position']
            seq_position += [0] * (max(seq_length) - l)
            seq.append(encode)
            posi.append(seq_position)
            mask.append([0] * l + [1] * (max(seq_length) - l))

        label = [element['code'] for element in batch]

        seq = torch.tensor(seq, requires_grad=False).cuda()
        label = torch.tensor(label, requires_grad=False).cuda()
        mask = torch.tensor(mask, requires_grad=False).byte().cuda()
        posi = torch.tensor(posi, requires_grad=False).cuda()

        self.position += batch_size
        if (self.position + batch_size) > len(self.data):
            self.epoch_finish = True

        return seq, label, seq_length, mask, posi


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
    # first 0 for padding
    # [n * embedding_size]
    embedding = []
    embedding.append([0.] * embedding_size)
    spell_checker = SpellChecker(raw_embedding.keys(), word_vocab)
    vocab_correction = []
    for i, word in enumerate(word_vocab):
        if word in raw_embedding.keys():
            vocab_correction.append(word)
            embedding.append(raw_embedding[word])
        else:
            word_correct = spell_checker.correct_word(word)
            if word_correct in raw_embedding.keys():
                # print('Change %s to %s' % (word, word_correct))
                vocab_correction.append(word_correct)
                embedding.append(raw_embedding[word_correct])
            else:
                # print(word, 'not in HealthVec')
                vocab_correction.append(word)
                rand = list(np.random.uniform(min_e, max_e, embedding_size))
                embedding.append(rand)

    return embedding, vocab_correction


def get_elmo(word_vocab):
    # get elmo embedding
    # first 0 for padding
    # [n * embedding_size]
    elmo = Elmo(elmo_options, elmo_weights, 2, dropout=0)
    character_ids = batch_to_ids([word_vocab])
    embeddings = elmo(character_ids)
    emb_low = embeddings['elmo_representations'][0].squeeze(0)
    emb_high = embeddings['elmo_representations'][1].squeeze(0)

    embedding_size = embeddings['elmo_representations'][0].size(-1)
    padding = [0.] * embedding_size
    emb_low = torch.cat((torch.tensor(padding).unsqueeze(0), emb_low), dim=0)
    emb_high = torch.cat((torch.tensor(padding).unsqueeze(0), emb_high), dim=0)

    return emb_low, emb_high


class SpellChecker(object):

    def __init__(self, spell_checker_vocab, origin_vocab):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.model = collections.defaultdict(int)
        for word in spell_checker_vocab:
            token, occurrence = word, 1
            if word in origin_vocab:
                occurrence += 1
            self.model[token] = occurrence

    def edits1(self, word):
        s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in s if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in s for c in self.alphabet if b]
        inserts = [a + c + b for a, b in s for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.model)

    def known(self, words):
        return set(w for w in words if w in self.model)

    def correct_word(self, word):
        word = word.lower()
        candidates = self.known([word]) or \
                     self.known(self.edits1(word)) or \
                     self.known_edits2(word) or \
                     [word] if word.isalpha() else [word]
        return max(candidates, key=self.model.get)

    def correct_sequence(self, text):
        return [self.correct_word(token) for token in text]


if __name__ == "__main__":
    # read_file()
    word_vocab = read_vocab('data/wordls.txt')
    print(len(word_vocab))
    get_elmo(word_vocab)
