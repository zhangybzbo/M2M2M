
class Char_embed(nn.Module):
    '''character level embedding'''

    # TODO: unfinished
    def __init__(self, char_embed_size, num_char):
        super(Char_embed, self).__init__()

        self.char_embed_size = char_embed_size
        self.char_embed = nn.Embedding(num_char, char_embed_size, padding_idx=0)

        # convolutions of filters with different sizes
        self.convolutions = []
        # list of tuples: (the number of filter, width)
        self.filter_num_width = [(25, 1), (50, 2), (75, 3), (100, 4), (125, 5), (150, 6)]
        for out_channel, filter_width in self.filter_num_width:
            self.convolutions.append(
                nn.Conv2d(
                    1,  # in_channel
                    out_channel,  # out_channel
                    kernel_size=(char_embed_size, filter_width),  # (height, width)
                    bias=True
                )
            )

        self.highway_input_dim = sum([x for x, y in self.filter_num_width])

        self.batch_norm = nn.BatchNorm1d(self.highway_input_dim, affine=False)

        # highway net
        self.highway1 = Highway(self.highway_input_dim)
        self.highway2 = Highway(self.highway_input_dim)

    def conv_layers(self, x):
        chosen_list = list()
        for conv in self.convolutions:
            feature_map = nn.Tanh()(conv(x))  # (batch_size, out_channel, 1, max_word_len-width+1)
            chosen = torch.max(feature_map, 3)[0]  # (batch_size, out_channel, 1)
            chosen = chosen.squeeze()  # (batch_size, out_channel)
            chosen_list.append(chosen)

        return torch.cat(chosen_list, 1)  # (batch_size, total_num_filers)

    def forward(self, x):
        # input: (batch, s, l)
        # output: (batch, s, total_num_filters)
        batch_size = x.size()[0]
        seq_len = x.size()[1]

        x = x.contiguous().view(-1, x.size()[2])
        c_embed = self.char_embed(x)  # (batch * s, l, embed)

        c_embed = torch.transpose(c_embed.view(c_embed.size()[0], 1, c_embed.size()[1], -1), 2, 3)
        # (batch * s, 1, embed, l)

        conv_w = self.conv_layers(c_embed)  # (batch * s, total_num_filters)
        conv_w = self.batch_norm(conv_w)

        y = self.highway1(conv_w)
        y = self.highway2(y)

        y = y.contiguous().view(batch_size, seq_len, -1)

        return y