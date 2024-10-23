import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.optim import Adam, lr_scheduler

import tqdm
from datetime import date
import sys
import signal
from omegaconf import OmegaConf

from IPython.display import clear_output
import matplotlib.pyplot as plt

from src.model import Text2Emoji
from src.parser import Text2EmojiParser
from src.dataset import Text2EmojiDataset
from src.utils import print_model, get_glove_embbedings, seed_all


def evaluate_loss_test(model, test_data_loader, loss, emoji_vocab_size):
    mean_loss = 0
    model.eval()

    with torch.no_grad():
        for batch in test_data_loader:
            batch_en_ids = batch['en_ids']
            batch_de_ids = batch['de_ids']

            logits = model(batch_en_ids, batch_de_ids)
            loss_t = loss(logits, one_hot(batch_de_ids.permute(1, 0)[:, 1:],
                                          num_classes=emoji_vocab_size).to(torch.float))

            mean_loss += loss_t.item()

    return mean_loss / len(test_data_loader)


def print_learn_curve(history):
    clear_output(True)
    plt.close('all')
    plt.figure(figsize=(12, 4))
    for i, (name, h) in enumerate(sorted(history.items())):
        plt.subplot(1, len(history), i + 1)
        plt.title(name)
        plt.plot(*zip(*h))
        plt.grid()
    plt.savefig(f'./data/learning_curves/curve_{date.today()}')
    # plt.show()


def train_model(model, dataset, n_epoch, print_step, emoji_vocab_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device=torch.device(device))
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 7], gamma=0.464159)
    loss = CrossEntropyLoss()

    batch_milestones = [2, 4, 7]
    batch_sizes = [32, 64, 128, 256]
    batch_step = 0

    epoch_emb_requires_grad = 4

    history = {'train_loss': [], 'test_loss': []}
    train_loss = 0

    test_learn_curve_increases = 0

    train_data_loader, test_data_loader = dataset.get_data_loader(batch_sizes[0], pad_idx)
    for epoch in range(n_epoch):
        if epoch == epoch_emb_requires_grad:
            model.emb_requires_grad()

        if epoch in batch_milestones:
            train_data_loader, test_data_loader = dataset.get_data_loader(batch_sizes[batch_step], pad_idx)
            batch_step += 1
        batch_size = batch_sizes[batch_step]

        print(f'epoch: {epoch + 1}/{n_epoch}, '
              f'lr: {scheduler.get_last_lr()}, '
              f'batch_size: {batch_size}')
        for i, batch in tqdm.tqdm(enumerate(train_data_loader)):
            model.train()

            batch_en_ids = batch['en_ids']
            batch_de_ids = batch['de_ids']

            optimizer.zero_grad()

            logits = model(batch_en_ids, batch_de_ids)
            loss_t = loss(logits, one_hot(batch_de_ids.permute(1, 0)[:, 1:],
                                          num_classes=emoji_vocab_size).to(torch.float))
            loss_t.backward()
            optimizer.step()

            train_loss += loss_t.item()
            if i % print_step == 0 and i != 0:
                model.eval()

                # evaluate
                mean_train_loss = train_loss / print_step
                train_loss = 0
                mean_test_loss = evaluate_loss_test(model, test_data_loader, loss, emoji_vocab_size)
                print(f'step: {i}/{n_epoch * len(train_data_loader)}, '
                      f'train_loss: {mean_train_loss}, '
                      f'test_loss: {mean_test_loss}')
                history['train_loss'].append((i, mean_train_loss))
                history['test_loss'].append((i, mean_test_loss))

                # save state
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'batch_size': batch_size,
                    'loss': loss
                }, f'./data/checkpoints/checkpoint_{date.today()}.pth')

                # plot learning curve
                print_learn_curve(history)

                # callbacks
                if len(history['test_loss']) > 1 and history['test_loss'][-2][1] < history['test_loss'][-1][1]:
                    test_learn_curve_increases += 1
                else:
                    test_learn_curve_increases = 0

                if test_learn_curve_increases > 5:
                    return history
        scheduler.step()

    return history


def load_model(model, path='data/checkpoints/checkpoint_2024-10-22.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])

    return model


if __name__ == '__main__':
    # set configs
    model_config = OmegaConf.load('./configs/model.yaml')
    train_config = OmegaConf.load('./configs/train.yaml')
    processing_config = OmegaConf.load('./configs/processing.yaml')

    st = processing_config.special_tokens
    pad_token, sos_token, eos_token, unk_token = st.pad.token, st.sos.token, st.eos.token, st.unk.token
    pad_idx, sos_idx, eos_idx, unk_idx = st.pad.id, st.sos.id, st.eos.id, st.unk.id

    seed_all(train_config.seed)

    # prepare data
    parser = Text2EmojiParser(pad_token=pad_token, sos_token=sos_token, eos_token=eos_token, unk_token=unk_token)
    dataset = Text2EmojiDataset()

    dataset.download_and_tokenization_dataset(parser.tokenize_emoji, parser.tokenize_text,
                                              processing_config.data.max_text_length, train_config.seed)

    parser.create_vocab(dataset.dataset['tokenized_emoji'],
                        dataset.dataset['tokenized_text'],
                        processing_config.data.min_freq_emoji,
                        processing_config.data.min_freq_text)
    parser.set_default_index(unk_idx)

    dataset.numericalize_dataset(parser.numericalize_data)
    dataset.train_test_split(processing_config.data.train_test_ratio)

    # create model and train
    embbedings, embbeding_size = get_glove_embbedings(parser.text_vocab.get_itos()[1:])

    model = Text2Emoji(parser.text_vocab_size(), parser.emoji_vocab_size(), sos_idx, eos_idx, pad_idx, embbeding_size,
                       model_config.model_architecture.hidden_size,
                       model_config.model_architecture.num_layers,
                       model_config.model_architecture.dropout,
                       model_config.model_architecture.sup_unsup_ratio)
    model.init_en_emb(embbedings)
    print_model(model)

    model = load_model(model)


    def signal_capture(sig, frame):
        torch.save(model.state_dict(), f'./data/saved_models/SIGINT_model_weights_{date.today()}.pth')
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_capture)

    train_history = train_model(model, dataset,
                                train_config.train_process.epoch,
                                train_config.train_process.print_step,
                                parser.emoji_vocab_size())

    torch.save(model.state_dict(), f'./data/saved_models/trained_model_weights_{date.today()}.pth')
