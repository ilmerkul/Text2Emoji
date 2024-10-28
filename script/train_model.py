import torch

import sys
import signal
from datetime import date
from omegaconf import OmegaConf
from loguru import logger
from argparse import ArgumentParser

from src.model import Text2Emoji
from src.parser import Text2EmojiParser
from src.dataset import Text2EmojiDataset
from src.utils import print_model, seed_all, train_model, set_logger

if __name__ == '__main__':
    # set logger
    set_logger()

    # set argparse
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--model_config', required=False, default='../configs/model.yaml')
    arg_parser.add_argument('--train_config', required=False, default='../configs/train.yaml')
    arg_parser.add_argument('--processing_config', required=False, default='../configs/processing.yaml')
    arg_parser.add_argument('--path_load', required=False, default='../data/datasets')
    arg_parser.add_argument('--path_save_model', required=False, default='./data/saved_models')
    args = arg_parser.parse_args()

    path_model_config = args.model_config
    path_train_config = args.train_config
    path_processing_config = args.processing_config
    path_load = args.path_load
    path_save_model = args.path_save_model

    # set configs
    model_config = OmegaConf.load(path_model_config)
    train_config = OmegaConf.load(path_train_config)
    processing_config = OmegaConf.load(path_processing_config)

    st = processing_config.special_tokens
    pad_token, sos_token, eos_token, unk_token = st.pad.token, st.sos.token, st.eos.token, st.unk.token
    pad_idx, sos_idx, eos_idx, unk_idx = st.pad.id, st.sos.id, st.eos.id, st.unk.id

    seed_all(train_config.seed)

    dataset = Text2EmojiDataset()
    dataset.load(path_load)

    parser = Text2EmojiParser(pad_token, sos_token, eos_token, unk_token)
    parser.load(f'{path_load}/parser/parser.pth')

    embbedings = torch.load(f'{path_load}/transfer/embbedings.pt')
    embbeding_size = embbedings.shape[1]

    logger.info('Model creating')
    model = Text2Emoji(parser.text_vocab_size(), parser.emoji_vocab_size(),
                       sos_idx, eos_idx, pad_idx, embbeding_size,
                       model_config.model_architecture.hidden_size,
                       model_config.model_architecture.num_layers,
                       model_config.model_architecture.dropout,
                       model_config.model_architecture.sup_unsup_ratio)
    model.init_en_emb(embbedings)
    print_model(model)


    def signal_capture(sig, frame):
        torch.save(model.state_dict(), f'{path_save_model}/SIGINT_model_weights_{date.today()}.pth')
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_capture)

    logger.info('Model training')
    train_history = train_model(model, dataset,
                                train_config.train_process.epoch,
                                train_config.train_process.print_step,
                                parser.emoji_vocab_size(), pad_idx)

    torch.save(model.state_dict(), f'{path_save_model}/trained_model_weights_{date.today()}.pth')
