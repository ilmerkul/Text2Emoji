from datasets import load_dataset

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

import gensim.downloader as api

import tqdm
from datetime import date
import sys
import signal
from omegaconf import OmegaConf

from src.model import Text2Emoji
from src.parser import Text2EmojiParser
from src.utils import print_model


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def check_none(row):
    return row['text'] is not None and row['emoji'] is not None and row['topic'] is not None


def download_and_tokenization_dataset(parser, max_text_length, seed=42):
    dataset = load_dataset('KomeijiForce/Text2Emoji', split='train')
    dataset.shuffle(seed=seed)
    dataset = dataset.filter(check_none)

    dataset = dataset.map(parser.tokenize_emoji, num_proc=torch.cpu.device_count())
    dataset = dataset.map(parser.tokenize_text, fn_kwargs={'max_length': max_text_length},
                          num_proc=torch.cpu.device_count())

    return dataset


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


def get_glove_embbedings(vocab):
    word_vectors = api.load("glove-wiki-gigaword-100")

    embbedings = []
    embbeding_size = 100
    # pad
    embbedings.append(np.zeros(embbeding_size))

    glove_word_count = 0
    for word in vocab:
        if word_vectors.has_index_for(word):
            embbedings.append(word_vectors[word])
            glove_word_count += 1
        else:
            embbedings.append(
                np.random.uniform(-1 / np.sqrt(embbeding_size), 1 / np.sqrt(embbeding_size), embbeding_size))

    print(f'glove_word_count: {glove_word_count}, size of vocab: {len(vocab)}')

    embbedings = torch.tensor(embbedings, dtype=torch.float32)

    return embbedings, embbeding_size


def evaluate_loss_test(model, test_data_loader, loss, emoji_vocab_size):
    mean_loss = 0
    model.eval()
    # sent = [1, 6, 2]
    # print(text_vocab.lookup_tokens(sent))
    # print(emoji_vocab.lookup_tokens(model.translate(torch.tensor(sent).unsqueeze(1), 16).squeeze().tolist()))
    with torch.no_grad():
        for batch in test_data_loader:
            batch_en_ids = batch['en_ids']
            batch_de_ids = batch['de_ids']

            logits = model(batch_en_ids, batch_de_ids)

            loss_t = loss(logits,
                          F.one_hot(batch_de_ids.permute(1, 0)[:, 1:], num_classes=emoji_vocab_size).to(torch.float))

            mean_loss += loss_t.item()

    return mean_loss / len(train_data_loader)


def train_model(model, train_data_loader, n_epoch, print_step, emoji_vocab):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device=torch.device(device))
    optimazer = Adam(model.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()

    history_loss = []
    for epoch in range(n_epoch):
        print(f'epoch: {epoch} / {n_epoch}')
        for i, batch in tqdm.tqdm(enumerate(train_data_loader)):
            model.train()

            batch_en_ids = batch['en_ids']
            batch_de_ids = batch['de_ids']

            optimazer.zero_grad()

            logits = model(batch_en_ids, batch_de_ids)

            loss_t = loss(logits,
                          F.one_hot(batch_de_ids.permute(1, 0)[:, 1:], num_classes=emoji_vocab).to(torch.float))

            loss_t.backward()

            optimazer.step()

            history_loss.append(loss_t.item())

            if i % print_step == 0 and i != 0:
                model.eval()
                batch_size = batch_en_ids.shape[1]
                mean_train_loss = sum(history_loss[(i - print_step):i]) / (print_step * batch_size)
                mean_test_loss = evaluate_loss_test(model, test_data_loader, loss, emoji_vocab)
                print(
                    f'step: {i} / {n_epoch * len(train_data_loader)}, train_loss: {mean_train_loss}, test_loss: {mean_test_loss}')

                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optim': optimazer.state_dict(),
                    'loss': loss
                }, f'./data/checkpoints/checkpoint_{date.today()}.pth')

    return history_loss


if __name__ == '__main__':
    # set configs
    model_config = OmegaConf.load('./configs/model.yaml')
    train_config = OmegaConf.load('./configs/train.yaml')
    processing_config = OmegaConf.load('./configs/processing.yaml')

    seed_all(train_config.seed)

    st = processing_config.special_tokens
    pad_token, sos_token, eos_token, unk_token = st.pad.token, st.sos.token, st.eos.token, st.unk.token
    pad_idx, sos_idx, eos_idx, unk_idx = st.pad.id, st.sos.id, st.eos.id, st.unk.id

    # prepare data
    parser = Text2EmojiParser(pad_token=pad_token, sos_token=sos_token, eos_token=eos_token, unk_token=unk_token)

    dataset = download_and_tokenization_dataset(parser, processing_config.data.max_text_length, train_config.seed)

    parser.create_vocab(dataset['tokenized_emoji'], dataset['tokenized_text'],
                        processing_config.data.min_freq_emoji,
                        processing_config.data.min_freq_text)
    parser.emoji_vocab.set_default_index(unk_idx)
    parser.text_vocab.set_default_index(unk_idx)

    data_type = "torch"
    format_columns = ["emoji_ids", "text_ids"]
    dataset = dataset.map(parser.numericalize_data)
    dataset = dataset.with_format(type=data_type, columns=format_columns, output_all_columns=True)
    print(dataset)

    dataset = dataset.train_test_split(test_size=processing_config.data.train_test_ratio)
    train_dataset, test_dataset = (dataset['train'], dataset['test'])
    train_data_loader = get_data_loader(train_dataset, train_config.train_process.batch_size, pad_idx, shuffle=True)
    test_data_loader = get_data_loader(test_dataset, train_config.train_process.batch_size, pad_idx)
    print(f'train_size: {len(train_data_loader)}, test_size: {len(test_data_loader)}')

    # create model and train
    embbedings, embbeding_size = get_glove_embbedings(parser.text_vocab.get_itos()[1:])

    model = Text2Emoji(parser.text_vocab_size(), parser.emoji_vocab_size(), sos_idx, eos_idx, pad_idx, embbeding_size,
                       model_config.model_architecture.hidden_size,
                       model_config.model_architecture.num_layers,
                       model_config.model_architecture.dropout,
                       model_config.model_architecture.sup_unsup_ratio)
    model.init_en_emb(embbedings)
    print_model(model)


    # checkpoint = torch.load('data/checkpoints/checkpoint_2024-10-21.pth')
    # model.load_state_dict(checkpoint['model'])

    def signal_capture(sig, frame):
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_capture)

    history_loss = train_model(model, train_data_loader,
                               train_config.train_process.epoch,
                               train_config.train_process.print_step,
                               parser.emoji_vocab_size())

    signal.pause()

    torch.save(model.state_dict(), './data/saved_models/model_weights.pth')
