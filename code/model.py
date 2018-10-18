import torch
import torch.nn as nn
from torch.autograd import Variable


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


class SoftDotAttention(nn.Module):

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class AttnNet(nn.Module):
    def __init__(self, pretrained_embed, embedding_size, hidden_size, output_size, dropout_ratio=0.5, num_layers=1,
                 bidirectional=True):
        super(AttnNet, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding.from_pretrained(pretrained_embed)
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
        '''
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
        weight_ctx = torch.cat((attn_combine, h_t), 1) # batch x 2h'''
        ctx_out = Variable(torch.zeros(0)).cuda()
        for s in range(ctx.size(1)):
            h_attn, _ = self.attn(ctx[:, s, :], ctx, mask)
            ctx_out = torch.cat((ctx_out, h_attn.unsqueeze(1)), 1)

        ctx_out = torch.mean(ctx_out, 1)

        ctx_out = nn.Tanh()(self.out(ctx_out))
        output = self.BN(ctx_out)
        # output = self.softmax(ctx_out)
        return output
