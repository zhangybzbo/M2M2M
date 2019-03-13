import numpy as np
import re
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import collections
import random
import BioBERT_token

random.seed(1)

file_list = ['AskAPatient/AskAPatient.fold-0.test.txt', 'AskAPatient/AskAPatient.fold-0.train.txt', 'AskAPatient/AskAPatient.fold-0.validation.txt']
file_list_2 = ['TwADR-L/TwADR-L.fold-0.test.txt', 'TwADR-L/TwADR-L.fold-0.train.txt', 'TwADR-L/TwADR-L.fold-0.validation.txt']
file_dir = 'datasets/'
save_code = 'codels.txt'
save_word = 'wordls.txt'
save_char = 'charls.txt'
save_code_2 = 'codels_2.txt'
save_word_2 = 'wordls_2.txt'
elmo_options = 'models/elmo_2x4096_512_2048cnn_2xhighway_options.json'
elmo_weights = 'models/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'


def read_file():
    '''get code & word list, dump into two files'''
    fw = open(file_dir + save_code, "w")
    fw_2 = open(file_dir + save_word, 'w')
    code_list = []
    word_list = []
    for f in file_list:
        with open(file_dir + f, encoding='windows-1252') as fr:
            for line in fr.readlines():
                '''# code = line.strip().split('\t')[1]
                # phrase = line.strip().split('\t')[0]
                # words = re.split(' |,|\)|\(|-|/|\.|\'|\"', phrase.strip())
                code = line.strip().split('\t')[2]
                phrase = line.strip().split('\t')[1]
                words = re.split(' |,|\)|\(|-|/|\.|\'|\"|\[|\]|\\\\', phrase.strip())
                if code not in code_list:
                    code_list.append(code)
                    fw.write(code + '\n')
                for word in words:
                    if (word.strip().lower() not in word_list) and (not word.strip() == ''):
                        word_list.append(word.strip().lower())
                        fw_2.write(word.lower() + '\n')'''
                code = line.strip().split('\t')[0]
                if code not in code_list:
                    code_list.append(code)
                    fw.write(code + '\n')
                phrase = line.strip().split('\t')[2]
                words = re.split(' |,|\)|\(|-|/|\.|\'|\"', phrase.strip())
                for word in words:
                    if (word.strip().lower() not in word_list) and (not word.strip() == ''):
                        word_list.append(word.strip().lower())
                        fw_2.write(word.lower() + '\n')

    fw.close()
    fw_2.close()
    print(len(code_list))
    print(len(word_list))


def get_character(wls):
    '''get character list from word list'''
    fw = open(file_dir + save_char, "w")
    char_ls = []
    for word in wls:
        for c in word.strip():
            if c not in char_ls:
                char_ls.append(c)
                fw.write(c + '\n')

    print(char_ls)
    fw.close()


def word_to_index(vocab):
    ''' word list to index
        {'word': index}'''
    list_to_index = {}
    for i, word in enumerate(vocab):
        list_to_index[word] = i + 1
        # for word list, 0=padding

    return list_to_index


def code_to_index(vocab):
    ''' code list to index
        {'code': index}'''
    list_to_index = {}
    for i, word in enumerate(vocab):
        list_to_index[word] = i

    return list_to_index


def char_to_index(vocab):
    ''' character list to index
        {'character': index}'''
    list_to_index = {}
    for i, word in enumerate(vocab):
        list_to_index[word] = i + 1
        # for character list, 0=padding

    return list_to_index


def read_vocab(path):
    ''' read list from file
        [n * 'word']'''
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


class tokenizer(object):
    ''' token data
        data: [ n * {'phrase': [word_id_list], 'length': int, 'emb_length': int,
            'emb': tensor(standard word embeddings), 'position': [word position in a sentence], 'code': code_id} ]'''

    def __init__(self, wordls, codels, datafile, pretrain_type=None):
        self.data = []
        self.pretrain = pretrain_type
        if pretrain_type == 'elmo_repre':
            self.pre_model = Elmo(elmo_options, elmo_weights, 2, dropout=0).cuda()
        elif pretrain_type == 'elmo_layer':
            self.pre_model = ElmoEmbedder(elmo_options, elmo_weights, cuda_device=0)
        elif pretrain_type == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.pre_model = BertModel.from_pretrained('bert-base-uncased').cuda()
            self.pre_model.eval()
        elif pretrain_type == 'biobert':
            tokenizer = BioBERT_token.FullTokenizer(vocab_file='models/pubmed_pmc_470k/vocab.txt', do_lower_case=True)
            self.pre_model = BertModel.from_pretrained('models/pubmed_pmc_470k/', from_tf=True).cuda()
            self.pre_model.eval()


        #with open(datafile, encoding='windows-1252') as f:
        with open(datafile) as f:
            '''for line in f.readlines():
                new_data = dict()
                #code = line.strip().split('\t')[1]
                #phrase = line.strip().split('\t')[0]
                #words = re.split(' |,|\)|\(|-|/|\.|\'|\"', phrase.strip())
                code = line.strip().split('\t')[2]
                phrase = line.strip().split('\t')[1]
                words = re.split(' |,|\)|\(|-|/|\.|\'|\"|\[|\]|\\\\', phrase.strip())
                words = [word.strip() for word in words if not word.strip() == '']
                wordtok = [wordls[word] for word in words]

                new_data['phrase'] = wordtok
                new_data['length'] = len(wordtok)
                new_data['emb_length'] = len(wordtok)  # for bert when using rare word

                if pretrain_type == 'elmo_repre':
                    elmo_id = batch_to_ids([words]).cuda()
                    pre_embed = self.pre_model(elmo_id)
                    new_data['emb'] = pre_embed['elmo_representations'][1].squeeze(0).detach()
                elif pretrain_type == 'elmo_layer':
                    pre_embed = self.pre_model.embed_sentence(words)
                    new_data['emb'] = torch.zeros((3, len(wordtok), 1024), requires_grad=False)
                    for i in range(3):
                        new_data['emb'][i, :, :] = torch.from_numpy(pre_embed[i])
                elif pretrain_type == 'bert':
                    bert_text = tokenizer.tokenize(' '.join(words))
                    bert_token = tokenizer.convert_tokens_to_ids(bert_text)
                    bert_tensor = torch.tensor([bert_token]).cuda()
                    new_data['emb_length'] = len(bert_token)
                    pre_embed, _ = self.pre_model(bert_tensor)
                    new_data['emb'] = pre_embed[11].squeeze(0).detach()

                new_data['position'] = [i + 1 for i in range(new_data['emb_length'])]
                new_data['code'] = codels[code]

                self.data.append(new_data)'''

            for line in f.readlines():

                code = line.strip().split('\t')[1]
                phrase = line.strip().split('\t')[0]
                new_data = dict()
                words = re.split(' |,|\)|\(|-|/|\.|\'|\"', phrase.strip())
                words = [word.strip().lower() for word in words if not word.strip() == '']
                wordtok = [wordls[word] for word in words]

                new_data['phrase'] = wordtok
                new_data['length'] = len(wordtok)
                new_data['emb_length'] = len(wordtok)  # for bert when using rare word

                if pretrain_type == 'elmo_repre':
                    elmo_id = batch_to_ids([words]).cuda()
                    pre_embed = self.pre_model(elmo_id)
                    new_data['emb'] = pre_embed['elmo_representations'][1].squeeze(0).detach()
                elif pretrain_type == 'elmo_layer':
                    pre_embed = self.pre_model.embed_sentence(words)
                    new_data['emb'] = torch.zeros((3, len(wordtok), 1024), requires_grad=False)
                    for i in range(3):
                        new_data['emb'][i, :, :] = torch.from_numpy(pre_embed[i])
                elif pretrain_type == 'bert':
                    bert_text = tokenizer.tokenize(' '.join(words))
                    bert_text = ['[CLS]'] + bert_text + ['[SEP]']
                    bert_token = tokenizer.convert_tokens_to_ids(bert_text)
                    bert_tensor = torch.tensor([bert_token]).cuda()
                    new_data['emb_length'] = len(bert_token)
                    pre_embed, _ = self.pre_model(bert_tensor)
                    new_data['emb'] = pre_embed[-2].squeeze(0).detach()
                elif pretrain_type == 'biobert':
                    bert_text = [tokenizer.tokenize(word) for word in words]
                    bert_text = ['[CLS]'] + bert_text + ['[SEP]']
                    bert_token = tokenizer.convert_tokens_to_ids(bert_text)
                    bert_tensor = torch.tensor([bert_token]).cuda()
                    new_data['emb_length'] = len(bert_token)
                    pre_embed, _ = self.pre_model(bert_tensor)
                    new_data['emb'] = pre_embed[-2].squeeze(0).detach()

                new_data['position'] = [i + 1 for i in range(new_data['emb_length'])]
                new_data['code'] = codels[code]

                self.data.append(new_data)

        self.max_length = max([l['emb_length'] for l in self.data])

        self.epoch_finish = False
        self.position = 0

    def reset_epoch(self):
        self.epoch_finish = False
        self.position = 0
        random.shuffle(self.data)

    def get_batch(self, batch_size):
        batch = self.data[self.position:self.position + batch_size]
        seq = []
        seq_length = [element['emb_length'] for element in batch]
        mask = []
        posi = []

        if self.pretrain == 'elmo_repre':
            pre_model = torch.zeros((batch_size, max(seq_length), 1024), requires_grad=False).cuda()
        elif self.pretrain == 'elmo_layer':
            pre_model = torch.zeros((batch_size, 3, max(seq_length), 1024), requires_grad=False).cuda()
        elif self.pretrain == 'bert' or self.pretrain == 'biobert':
            pre_model = torch.zeros((batch_size, max(seq_length), 768), requires_grad=False).cuda()
        else:
            pre_model = None

        for i, element in enumerate(batch):
            if self.pretrain == 'elmo_repre' or self.pretrain == 'bert' or self.pretrain == 'biobert':
                pre_model[i, :element['emb_length'], :] = element['emb']
            elif self.pretrain:
                pre_model[i, :, :element['emb_length'], :] = element['emb']
            seq.append(element['phrase'] + [0] * (max(seq_length) - element['length']))
            posi.append(element['position'] + [0] * (max(seq_length) - element['emb_length']))
            mask.append([0] * element['emb_length'] + [1] * (max(seq_length) - element['emb_length']))

        label = [element['code'] for element in batch]

        seq = torch.tensor(seq, requires_grad=False).cuda()
        label = torch.tensor(label, requires_grad=False).cuda()
        mask = torch.tensor(mask, requires_grad=False).byte().cuda()
        posi = torch.tensor(posi, requires_grad=False).cuda()

        self.position += batch_size
        if (self.position + batch_size) > len(self.data):
            self.epoch_finish = True

        return seq, label, seq_length, mask, posi, pre_model


def read_embed(path):
    ''' read embedding from file
        {'word': [embedding]}'''
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


def pre_embed(raw_embedding, word_vocab, max_e, min_e, embedding_size, spell_check=False):
    ''' permute embedding to word id
        first 0 for padding
        [n * embedding_size]'''
    embedding = []
    embedding.append([0.] * embedding_size)
    if spell_check:
        spell_checker = SpellChecker(raw_embedding.keys(), word_vocab)
        vocab_correction = []
    for i, word in enumerate(word_vocab):
        if word in raw_embedding.keys():
            if spell_check:
                vocab_correction.append(word)
            embedding.append(raw_embedding[word])
        else:
            if spell_check:
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
            else:
                rand = list(np.random.uniform(min_e, max_e, embedding_size))
                embedding.append(rand)

    if spell_check:
        return embedding, vocab_correction
    else:
        return embedding


def get_elmo(word_vocab):
    ''' get elmo embedding, character level base, without context
        first 0 for padding
        [n * embedding_size]'''
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
    read_file()
    word_vocab = read_vocab('datasets/wordls_2.txt')
    print(len(word_vocab))
    code_vocab = read_vocab('datasets/codels_2.txt')
    print(len(code_vocab))
    # get_character(word_vocab)
