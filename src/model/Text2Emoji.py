import torch
from torch import nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Attention class for Text2Emoji model."""

    def __init__(self, hid_size: int):
        super(AttentionLayer, self).__init__()

        self.hid_size = hid_size
        self.linear0 = nn.Linear(2 * hid_size, hid_size)
        self.act = nn.Tanh()
        self.linear1 = nn.Linear(hid_size, 1)

    def forward(self, hidden: torch.Tensor, enc_seq: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # hidden (batch_size, hid_size)
        # enc_seq (batch_size, source_length, hid_size)
        # mask (batch_size, source_length)
        batch_size = enc_seq.shape[0]
        source_length = enc_seq.shape[1]

        # rep_hidden (batch_size, source_length, hid_size)
        rep_hidden = hidden.repeat(1, 1, source_length).reshape(batch_size,
                                                                source_length,
                                                                self.hid_size)
        # pre_scores (batch_size, source_length, 2 * hid_size)
        pre_scores = torch.concatenate([enc_seq, rep_hidden], dim=-1)

        # scores (batch_size, source_length)
        scores = self.linear1(self.act(self.linear0(pre_scores))).squeeze()

        # scores (batch_size, source_length)
        scores = torch.where(mask, scores, float("-inf"))

        # attention_weights (batch_size, source_length)
        attention_weights = F.softmax(scores, dim=-1)

        # new_hidden (batch_size, hid_size)
        new_hidden = torch.bmm(attention_weights.unsqueeze(-2),
                               enc_seq).squeeze()

        return new_hidden


class Encoder(nn.Module):
    """Encoder class for Text2Emoji model."""

    def __init__(self, en_vocab_size: int, emb_size: int, pad_id: int,
                 hid_size: int, num_layers: int, dropout: float):
        super(Encoder, self).__init__()

        self.emb_size = emb_size
        self.hid_size = hid_size

        self.emb = nn.Embedding(en_vocab_size, emb_size, padding_idx=pad_id)
        self.enc0 = nn.GRU(emb_size, hid_size, num_layers=num_layers,
                           bias=True, batch_first=True, dropout=dropout,
                           bidirectional=True)
        self.hid_lin = nn.Linear(2 * hid_size, hid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (batch_size, source_length)
        x = self.emb(x)  # (batch_size, source_length, emb_size)

        # enc_seq (batch_size, source_length, 2 * hid_size)
        # _ (batch_size, 2 * hid_size)
        enc_seq, _ = self.enc0(x)
        # enc_seq (batch_size, source_length, hid_size)
        enc_seq = self.hid_lin(enc_seq)

        return enc_seq

    def init_emb(self, embbedings: torch.Tensor) -> None:
        """
        Set embbedings weights with requires_grad=False.
        :param embbedings: embbedings weights
        :return: None
        """
        self.emb.weight = nn.Parameter(embbedings, requires_grad=False)

    def emb_requires_grad(self) -> None:
        """
        Set requires_grad=True for model's embbedings.
        :return: None
        """
        self.emb.requires_grad_(requires_grad=True)


class Decoder(nn.Module):
    """Decoder class for Text2Emoji model."""

    def __init__(self, de_vocab_size: int, emb_size: int,
                 pad_id: int, hid_size: int, num_layers: int):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.hid_size = hid_size

        self.emb = nn.Embedding(de_vocab_size, emb_size, padding_idx=pad_id)
        self.dec0 = nn.GRUCell(emb_size, hid_size, bias=True)

        self.linear0 = nn.Linear(hid_size, de_vocab_size)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) \
            -> list[torch.Tensor, torch.Tensor]:
        # x (batch_size, )
        # hidden_state (batch_size, hid_size)

        # x (batch_size, emb_size)
        x = self.emb(x)

        # hidden_state (batch_size, hid_size)
        hidden_state = self.dec0(x, hidden_state)

        # logits (batch_size, de_vocab_size)
        logits = self.linear0(hidden_state)

        # return [(batch_size, hid_size), (batch_size, de_vocab_size)]
        return [hidden_state, logits]


class Text2Emoji(nn.Module):
    """Text2Emoji model"""

    def __init__(self, en_vocab_size: int, de_vocab_size: int, sos_id: int,
                 eos_id: int, pad_id: int, emb_size: int, hid_size: int,
                 num_layers: int, dropout: float = 0.2,
                 sup_unsup_ratio: float = 1.0):
        super(Text2Emoji, self).__init__()

        self.hid_size = hid_size
        self.pad_id = pad_id
        self.de_vocab_size = de_vocab_size
        self.sup_unsup_ratio = sup_unsup_ratio

        self.sos_id = sos_id
        self.eos_id = eos_id

        self.enc = Encoder(en_vocab_size, emb_size, pad_id, hid_size,
                           num_layers, dropout)
        self.dec = Decoder(de_vocab_size, emb_size, pad_id, hid_size,
                           num_layers)

        self.attention = AttentionLayer(hid_size)

    def forward(self, source_sent: torch.Tensor,
                target_sent: torch.Tensor) -> torch.Tensor:
        # source_sent (batch_size, source_length)
        # target_sent (batch_size, target_length)
        enc_seq = self.enc(source_sent)
        return self.decoder_forward(enc_seq, source_sent, target_sent)

    def decoder_forward(self, enc_seq: torch.Tensor, source_sent: torch.Tensor,
                        target_sent: torch.Tensor) -> torch.Tensor:
        # enc_seq (source_length, batch_size, hid_size)
        # source_sent (source_length, batch_size)
        # target_sent (target_length, batch_size)
        batch_size = enc_seq.shape[1]
        target_length = target_sent.shape[0]

        # enc_seq (batch_size, source_length, hid_size)
        enc_seq = enc_seq.permute(1, 0, 2)

        logits_sequence = []

        # mask (batch_size, source_length)
        mask = torch.where((source_sent == self.pad_id),
                           False,
                           True).permute(1, 0)
        lengths = ((source_sent != self.pad_id).to(torch.int64).sum(dim=0) - 1)
        mask.requires_grad = False
        lengths.requires_grad = False

        # state (batch_size, hid_size)
        state = enc_seq[torch.arange(batch_size), lengths]
        # logits # (batch_size, de_vocab_size)
        logits = F.one_hot(target_sent[0, :], num_classes=self.de_vocab_size)
        for i in range(target_length - 1):
            # target_pred (batch_size, )
            target_pred = torch.argmax(logits, dim=-1)
            target = target_sent[i, :]

            if torch.multinomial(torch.tensor([self.sup_unsup_ratio,
                                               1.0 - self.sup_unsup_ratio],
                                              dtype=torch.float), 1)[0]:
                target = target_pred

            # logits (batch_size, de_vocab_size)
            state, logits = self.dec(target, state)
            logits_sequence.append(logits)

            # calculate attention
            attention_state = self.attention(state, enc_seq, mask)

            state = state + attention_state

        logits_sequence = torch.stack(logits_sequence, dim=1)
        return logits_sequence

    def init_en_emb(self, embeddings: torch.Tensor) -> None:
        """
        Call encoder's init_emb function for set embbedings weight.
        :param embeddings: embbedings weights
        :return: None
        """
        self.enc.init_emb(embeddings)

    def emb_requires_grad(self) -> None:
        """
        Call encoder's emb_requires_grad function for embbedings
         tensor requires_grad=True.
        :return: None
        """
        self.enc.emb_requires_grad()

    def translate(self, source_sent: torch.Tensor,
                  max_length: int = 128) -> torch.Tensor:
        """
        Translate source sentence to emoji sentence.
        :param source_sent: text sentence
        :param max_length: maximum length of return sentence
        :return: target emoji sentence
        """
        # enc_seq (source_length, batch_size=1, hid_size)
        enc_seq = self.enc(source_sent)
        batch_size = enc_seq.shape[1]

        # enc_seq (batch_size=1, source_length, hid_size)
        enc_seq = enc_seq.permute(1, 0, 2)

        logits_sequence = []

        # mask (batch_size=1, source_length)
        mask = torch.where((source_sent == self.pad_id),
                           False,
                           True).permute(1, 0)
        lengths = ((source_sent != self.pad_id).to(torch.int64).sum(dim=0) - 1)
        # state (batch_size=1, hid_size)
        state = enc_seq[torch.arange(batch_size), lengths]

        # logits (batch_size=1, de_vocab_size)
        logits = F.one_hot(torch.full((batch_size,), self.sos_id),
                           num_classes=self.de_vocab_size)
        for i in range(max_length):
            # target (batch_size=1, )
            target = torch.argmax(logits, dim=-1)

            if target[0] == self.eos_id:
                break

            # logits (batch_size=1, de_vocab_size)
            state, logits = self.dec(target, state)
            logits_sequence.append(logits)

            # calculate attention
            attention_state = self.attention(state, enc_seq, mask)

            state = state + attention_state

        logits_sequence = torch.stack(logits_sequence, dim=1)
        return torch.argmax(logits_sequence, dim=-1)
