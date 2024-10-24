import torch

from datetime import date
import sys
import signal
from omegaconf import OmegaConf

from src.model import Text2Emoji
from src.parser import Text2EmojiParser
from src.dataset import Text2EmojiDataset
from src.utils import print_model, seed_all, train_model
from src.transfer import get_glove_embbedings


def load_model(m, path='data/checkpoints/checkpoint_2024-10-22.pth'):
    checkpoint = torch.load(path)
    m.load_state_dict(checkpoint['model'])

    return m


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


    def signal_capture(sig, frame):
        torch.save(model.state_dict(), f'./data/saved_models/SIGINT_model_weights_{date.today()}.pth')
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_capture)

    train_history = train_model(model, dataset,
                                train_config.train_process.epoch,
                                train_config.train_process.print_step,
                                parser.emoji_vocab_size(), pad_idx)

    torch.save(model.state_dict(), f'./data/saved_models/trained_model_weights_{date.today()}.pth')
