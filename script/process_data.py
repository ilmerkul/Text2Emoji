from torch import save

from omegaconf import OmegaConf
from loguru import logger

from src.parser import Text2EmojiParser
from src.dataset import Text2EmojiDataset
from src.utils import seed_all, set_logger
from src.transfer import get_glove_embbedings

from argparse import ArgumentParser

if __name__ == '__main__':
    # set logger
    set_logger()

    # set argparse
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--train_config', required=False, default='../configs/train.yaml')
    arg_parser.add_argument('--processing_config', required=False, default='../configs/processing.yaml')
    arg_parser.add_argument('--path_save', required=False, default='../data/datasets')
    args = arg_parser.parse_args()

    path_train_config = args.train_config
    path_processing_config = args.processing_config
    path_save = args.path_save

    # set configs
    train_config = OmegaConf.load(path_train_config)
    processing_config = OmegaConf.load(path_processing_config)

    st = processing_config.special_tokens
    pad_token, sos_token, eos_token, unk_token = st.pad.token, st.sos.token, st.eos.token, st.unk.token
    pad_idx, sos_idx, eos_idx, unk_idx = st.pad.id, st.sos.id, st.eos.id, st.unk.id

    seed_all(train_config.seed)

    # prepare data
    logger.info(f'Data preprocessing started with test size: {processing_config.data.train_test_ratio}')
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
    logger.info('Get Glove embbedings')
    embbedings, glove_word_count = get_glove_embbedings(parser.text_vocab.get_itos()[1:])
    logger.info(f'glove_word_count: {glove_word_count}, size of vocab: {len(parser.text_vocab.get_itos()) - 1}')

    parser.save(f'{path_save}/parser/parser.pth')
    save(embbedings, f"{path_save}/transfer/embbedings.pt")
    dataset.save(path_save)
