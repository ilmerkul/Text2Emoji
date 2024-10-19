from datasets import load_dataset

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchtext

import numpy as np

import emoji

from nltk.corpus import stopwords
import nltk
from nltk import WordPunctTokenizer, WordNetLemmatizer

from collections import Counter
import tqdm

from src.model import Text2Emoji

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokenizer = WordPunctTokenizer()
lemmatizer = WordNetLemmatizer()

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

MIN_FREQ_EMOJI = 5
MIN_FREQ_TEXT = 20
MAX_TEXT_LENGTH = 128
SPECIAL_TOKENS = {
    '<pad>': 0,
    '<sos>': 1,
    '<eos>': 2,
    '<unk>': 3
}

BATCH_SIZE = 128
EPOCH = 1
PRINT_STEP = 100


def tokenize_emoji(row, sos_token, eos_token):
    tokenized_row = list(map(lambda x: x if emoji.is_emoji(x) else '', row['emoji']))
    tokenized_row = [sos_token] + tokenized_row + [eos_token]
    row['tokenized_emoji'] = tokenized_row
    return row


def tokenize_text(row, max_length, sos_token, eos_token):
    tokenized_row = list(filter(lambda x: x not in stop_words and x.isalpha(),
                                map(lambda x: lemmatizer.lemmatize(x),
                                    nltk.word_tokenize(row['text'].lower())[:max_length])))
    tokenized_row = [sos_token] + tokenized_row + [eos_token]
    row['tokenized_text'] = tokenized_row
    return row


def numericalize_data(row, emoji_vocab, text_vocab):
    emoji_ids = emoji_vocab.lookup_indices(row['tokenized_emoji'])
    text_ids = text_vocab.lookup_indices(row['tokenized_text'])
    return {'emoji_ids': emoji_ids, 'text_ids': text_ids}


def check_none(row):
    return row['text'] is not None and row['emoji'] is not None and row['topic'] is not None


def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_en_ids = [example["text_ids"] for example in batch]
        batch_de_ids = [example["emoji_ids"] for example in batch]
        batch_en_ids = pad_sequence(batch_en_ids, padding_value=pad_index)
        batch_de_ids = pad_sequence(batch_de_ids, padding_value=pad_index)
        batch = {
            "en_ids": batch_en_ids,
            "de_ids": batch_de_ids,
        }
        return batch

    return collate_fn


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader


if __name__ == '__main__':
    dataset = load_dataset('KomeijiForce/Text2Emoji', split='train')
    dataset.shuffle(seed=SEED)
    dataset = dataset.filter(check_none)

    pad_token, sos_token, eos_token, unk_token = SPECIAL_TOKENS.keys()
    dataset = dataset.map(tokenize_emoji, fn_kwargs={'sos_token': sos_token, 'eos_token': eos_token},
                          num_proc=torch.cpu.device_count())
    dataset = dataset.map(tokenize_text,
                          fn_kwargs={'max_length': MAX_TEXT_LENGTH, 'sos_token': sos_token, 'eos_token': eos_token},
                          num_proc=torch.cpu.device_count())

    emoji_vocab = torchtext.vocab.build_vocab_from_iterator(dataset['tokenized_emoji'], min_freq=MIN_FREQ_EMOJI,
                                                            specials=list(SPECIAL_TOKENS.keys()))
    text_vocab = torchtext.vocab.build_vocab_from_iterator(dataset['tokenized_text'], min_freq=MIN_FREQ_TEXT,
                                                           specials=list(SPECIAL_TOKENS.keys()))

    unk_idx = SPECIAL_TOKENS[unk_token]
    emoji_vocab.set_default_index(unk_idx)
    text_vocab.set_default_index(unk_idx)
    print(emoji_vocab.get_itos()[:20])
    print(text_vocab.get_itos()[:20])

    data_type = "torch"
    format_columns = ["emoji_ids", "text_ids"]

    dataset = dataset.map(numericalize_data, fn_kwargs={'emoji_vocab': emoji_vocab, 'text_vocab': text_vocab})
    dataset = dataset.with_format(type=data_type, columns=format_columns, output_all_columns=True)

    # emoji_counter = Counter([token for sent in dataset['tokenized_emoji'] for token in sent.split()])
    # text_counter = Counter([token for sent in dataset['tokenized_text'] for token in sent.split()])
    # emoji_vocab = set([token for sent in dataset['tokenized_emoji'] for token in sent.split() if emoji_counter[token] > MIN_FREQ_EMOJI])
    # text_vocab = set([token for sent in dataset['tokenized_text'] for token in sent.split() if text_counter[token] > MIN_FREQ_TEXT])

    # print(len(emoji_vocab))
    # print(len(text_vocab))

    # emoji_token2id = {em: i for i, em in enumerate(emoji_vocab, start=len(SPECIAL_TOKENS))}
    # text_token2id = {t: i for i, t in enumerate(text_vocab, start=len(SPECIAL_TOKENS))}

    # for spec_tok, id_spec_tok in SPECIAL_TOKENS.items():
    #     emoji_token2id[spec_tok] = id_spec_tok
    #     text_token2id[spec_tok] = id_spec_tok

    dataset = dataset.train_test_split(test_size=0.2)

    train_dataset, test_dataset = (dataset['train'], dataset['test'])

    pad_index = SPECIAL_TOKENS[pad_token]
    train_data_loader = get_data_loader(train_dataset, BATCH_SIZE, pad_index, shuffle=True)
    test_data_loader = get_data_loader(test_dataset, BATCH_SIZE, pad_index)

    print(train_data_loader, test_data_loader)
    print(train_dataset)
    source_length, target_length = max(map(len, train_dataset['tokenized_text'])), max(
        map(len, train_dataset['tokenized_emoji']))

    model = Text2Emoji(len(text_vocab), len(emoji_vocab), 256, pad_index, 512, 1)
    optimazer = Adam(model.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()

    history_loss = []
    for epoch in range(EPOCH):
        print(f'epoch: {epoch} / {EPOCH}')
        for i, batch in tqdm.tqdm(enumerate(train_data_loader)):
            model.train()

            batch_en_ids = batch['en_ids']
            batch_de_ids = batch['de_ids']

            optimazer.zero_grad()

            logits = model(batch_en_ids, batch_de_ids)

            loss_t = loss(logits,
                          F.one_hot(batch_de_ids.permute(1, 0)[:, 1:], num_classes=len(emoji_vocab)).to(torch.float))

            loss_t.backward()

            optimazer.step()

            history_loss.append(loss_t.item())

            if i % PRINT_STEP == 0 and i != 0:
                model.eval()
                mean_loss = sum(history_loss[(i - PRINT_STEP):i]) / PRINT_STEP
                print(f'step: {i} / {EPOCH * len(train_data_loader)}, train_loss: {mean_loss}')
