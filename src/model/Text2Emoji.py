import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, en_vocab_size, emb_size, pad_id, hid_size, num_layers, dropout):
        super(Encoder, self).__init__()

        self.emb_size = emb_size
        self.hid_size = hid_size

        self.emb = nn.Embedding(en_vocab_size, emb_size, padding_idx=pad_id)
        self.enc0 = nn.GRU(emb_size, hid_size, num_layers=num_layers, bias=True, batch_first=True, dropout=dropout,
                           bidirectional=True)
        self.hid_lin = nn.Linear(2 * hid_size, hid_size)

    def forward(self, x):
        # x (batch_size, source_length)
        x = self.emb(x)  # (batch_size, source_length, emb_size)

        enc_seq, _ = self.enc0(x)  # (batch_size, source_length, 2 * hid_size) (batch_size, 2 * hid_size)
        enc_seq = self.hid_lin(enc_seq)  # (batch_size, source_length, hid_size)

        return enc_seq  # (batch_size, source_length, hid_size)


class Decoder(nn.Module):
    def __init__(self, de_vocab_size, emb_size, pad_id, hid_size, num_layers, dropout):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.hid_size = hid_size

        self.emb = nn.Embedding(de_vocab_size, emb_size, padding_idx=pad_id)
        self.dec0 = nn.GRUCell(emb_size, hid_size, bias=True)

        self.linear0 = nn.Linear(hid_size, de_vocab_size)

    def forward(self, x, hidden_state):
        # x (batch_size, )
        # hidden_state (batch_size, hid_size)

        x = self.emb(x)  # (batch_size, emb_size)

        hidden_state = self.dec0(x, hidden_state)  # (batch_size, hid_size)

        logits = self.linear0(hidden_state)  # (batch_size, de_vocab_size)

        return [hidden_state, logits]  # [(batch_size, hid_size), (batch_size, de_vocab_size)]


class Text2Emoji(nn.Module):
    def __init__(self, en_vocab_size, de_vocab_size, emb_size, pad_id, hid_size, num_layers, dropout=0.2):
        super(Text2Emoji, self).__init__()

        self.hid_size = hid_size
        self.pad_id = pad_id

        self.enc = Encoder(en_vocab_size, emb_size, pad_id, hid_size, num_layers, dropout)
        self.dec = Decoder(de_vocab_size, emb_size, pad_id, hid_size, num_layers, dropout)

    def forward(self, source_sent, target_sent):
        # source_sent (batch_size, source_length)
        # target_sent (batch_size, target_length)
        x = self.enc(source_sent)
        return self.decoder_forward(x, target_sent)

    def decoder_forward(self, x, target_sent):
        # x (batch_size, source_length, hid_size)
        # target_sent (target_length, batch_size)
        target_length = target_sent.shape[0]

        logits_sequence = []

        state = x[-1, :, :]
        for i in range(target_length - 1):
            state, logits = self.dec(target_sent[i, :], state)
            logits_sequence.append(logits)

        logits_sequence = torch.stack(logits_sequence, dim=1)
        return logits_sequence
