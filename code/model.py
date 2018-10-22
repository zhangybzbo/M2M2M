import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from submodel import SoftDotAttention, EncoderLayer


# without attention
class Network(nn.Module):
    def __init__(self, pretrained_embed, embedding_size, hidden_size, output_size, dropout_ratio=0.5, num_layers=1,
                 bidirectional=True):
        super(Network, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional)
        self.out = nn.Linear(hidden_size * self.num_directions, output_size)
        self.BN = nn.BatchNorm1d(output_size)  # size?
        # self.softmax = nn.LogSoftmax(dim=1)

    def init_state(self, inputs):
        # input size (seq, batch)
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return h0.cuda(), c0.cuda()

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        ctx, (h_t, c_t) = self.lstm(embeds, (h0, c0))
        ctx = torch.mean(ctx, 1)
        ctx_out = nn.Tanh()(self.out(ctx))
        output = self.BN(ctx_out)
        # output = self.softmax(output)
        return output


class AttnNet(nn.Module):
    def __init__(self, pretrained_embed, embedding_size, hidden_size, output_size, dropout_ratio=0.5, num_layers=1,
                 bidirectional=True):
        super(AttnNet, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional)
        self.attn = SoftDotAttention(hidden_size * self.num_directions)
        self.out = nn.Linear(hidden_size * self.num_directions, output_size)
        self.BN = nn.BatchNorm1d(output_size)  # size?
        # self.softmax = nn.LogSoftmax(dim=1)

    def init_state(self, inputs):
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return h0.cuda(), c0.cuda()

    def forward(self, inputs, mask=None):
        embeds = self.embedding(inputs)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)

        ctx, (h_t, c_t) = self.lstm(embeds, (h0, c0))
        
        ctx_out = Variable(torch.zeros(0)).cuda()
        for s in range(ctx.size(1)):
            h_attn, _ = self.attn(ctx[:, s, :], ctx, mask)
            ctx_out = torch.cat((ctx_out, h_attn.unsqueeze(1)), 1)

        ctx_out = torch.mean(ctx_out, 1)

        ctx_out = nn.Tanh()(self.out(ctx_out))
        output = self.BN(ctx_out)
        # output = self.softmax(ctx_out)
        return output

class HiddenNet(nn.Module):
    def __init__(self, pretrained_embed, embedding_size, hidden_size, output_size, dropout_ratio=0.5, num_layers=1,
                 bidirectional=True):
        super(HiddenNet, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional)
        self.attn = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions, bias=False)
        self.attn_sm = nn.Softmax(dim=1)
        self.out = nn.Linear(hidden_size * self.num_directions * 2, output_size)
        self.BN = nn.BatchNorm1d(output_size)  # size?
        # self.softmax = nn.LogSoftmax(dim=1)

    def init_state(self, inputs):
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return h0.cuda(), c0.cuda()

    def forward(self, inputs, mask=None):
        embeds = self.embedding(inputs)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)

        ctx, (h_t, c_t) = self.lstm(embeds, (h0, c0))
        # last h attention
        if self.num_directions == 2:
            h_t = torch.cat((h_t[-1], h_t[-2]), 1)
            c_t = torch.cat((c_t[-1], c_t[-2]), 1)
        else:
            h_t = h_t[-1]
            c_t = c_t[-1]  # (batch, hidden_size)
        attn_weight = self.attn(h_t).unsqueeze(2)   # b x h x 1
        attn_apply = torch.bmm(ctx, attn_weight).squeeze(2)
        if mask is not None:
            # -Inf masking prior to the softmax
            attn_apply.data.masked_fill_(mask, -float('inf'))
        attn_apply = self.attn_sm(attn_apply)
        attn_apply = attn_apply.view(attn_apply.size(0), 1, attn_apply.size(1))  # batch x 1 x s
        attn_combine = torch.bmm(attn_apply, ctx).squeeze(1)  # batch x h
        weight_ctx = torch.cat((attn_combine, h_t), 1) # batch x 2h
        
        ctx_out = nn.Tanh()(self.out(weight_ctx))
        output = self.BN(ctx_out)
        # output = self.softmax(ctx_out)
        return output

class TransformerNet(nn.Module):
    def __init__(self, pretrained_embed, len_max_seq, embedding_size, inner_hid_size, output_size, d_k, d_v,
                 dropout_ratio=0.1, num_layers=6, num_head=8):
        super(TransformerNet, self).__init__()
        self.n_position = len_max_seq + 1

        self.embedding = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
        self.pos_encode = nn.Embedding.from_pretrained(
            self.get_sinusoid_encoding_table(self.n_position, embedding_size, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(embedding_size, inner_hid_size, num_head, d_k, d_v, dropout=dropout_ratio)
            for _ in range(num_layers)])

        self.out = nn.Linear(embedding_size, output_size)
        self.BN = nn.BatchNorm1d(output_size)

    def forward(self, seq, seq_pos):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = self.get_attn_key_pad_mask(seq_k=seq, seq_q=seq)
        non_pad_mask = self.get_non_pad_mask(seq)

        # -- Forward
        enc_output = self.w_embedding(seq) + self.pos_encode(seq_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        ctx = torch.mean(enc_output, 1)

        ctx = nn.Tanh()(self.out(ctx))
        output = self.BN(ctx)

        return output

    def get_sinusoid_encoding_table(self, n_position, d_hid, padding_idx=None):
        ''' Sinusoid position encoding table '''

        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.

        return torch.FloatTensor(sinusoid_table).cuda()

    def get_non_pad_mask(self, seq):
        assert seq.dim() == 2
        return seq.ne(0).type(torch.float).unsqueeze(-1).cuda()

    def get_attn_key_pad_mask(self, seq_k, seq_q):
        ''' For masking out the padding part of key sequence. '''

        # Expand to fit the shape of key query attention matrix.
        len_q = seq_q.size(1)
        padding_mask = seq_k.eq(0)
        padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

        return padding_mask.cuda()
