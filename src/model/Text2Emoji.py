import torch
from torch import nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, hid_size):
        super(AttentionLayer, self).__init__()

        self.hid_size = hid_size
        self.linear0 = nn.Linear(2 * hid_size, hid_size)
        self.act = nn.Tanh()
        self.linear1 = nn.Linear(hid_size, 1)

    def forward(self, hidden, enc_seq, mask):
        # hidden (batch_size, hid_size)
        # enc_seq (batch_size, source_length, hid_size)
        # mask (batch_size, source_length)
        batch_size = enc_seq.shape[0]
        source_length = enc_seq.shape[1]

        pre_scores = torch.concatenate(
            [enc_seq, hidden.repeat(1, 1, source_length).reshape(batch_size, source_length, self.hid_size)],
            dim=-1)  # (batch_size, source_length, 2 * hid_size)

        scores = self.linear1(self.act(self.linear0(pre_scores))).squeeze()  # (batch_size, source_length)

        scores = torch.where(mask, scores, float('-inf'))  # (batch_size, source_length)

        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, source_length)

        new_hidden = torch.bmm(attention_weights.unsqueeze(-2), enc_seq).squeeze()  # (batch_size, hid_size)

        return new_hidden


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

        self.attention = AttentionLayer(hid_size)

    def forward(self, source_sent, target_sent):
        # source_sent (batch_size, source_length)
        # target_sent (batch_size, target_length)
        enc_seq = self.enc(source_sent)
        return self.decoder_forward(enc_seq, source_sent, target_sent)

    def decoder_forward(self, enc_seq, source_sent, target_sent):
        # enc_seq (source_length, batch_size, hid_size)
        # source_sent (source_length, batch_size)
        # target_sent (target_length, batch_size)
        batch_size = enc_seq.shape[1]
        sorce_length = source_sent.shape[0]
        target_length = target_sent.shape[0]

        enc_seq = enc_seq.permute(1, 0, 2)  # (batch_size, source_length, hid_size)

        logits_sequence = []

        # mask (batch_size, source_length)
        mask = torch.where((source_sent == self.pad_id), False, True).permute(1, 0)
        lengths = ((source_sent != self.pad_id).to(torch.int64).sum(dim=0) - 1)
        mask.requires_grad = False
        lengths.requires_grad = False
        state = enc_seq[torch.arange(batch_size), lengths]
        for i in range(target_length - 1):
            state, logits = self.dec(target_sent[i, :], state)
            logits_sequence.append(logits)

            # calculate attention
            attention_state = self.attention(state, enc_seq, mask)

            state = state + attention_state

        logits_sequence = torch.stack(logits_sequence, dim=1)
        return logits_sequence