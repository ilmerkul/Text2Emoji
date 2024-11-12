from typing import Iterable

import emoji
import nltk
from nltk import WordNetLemmatizer, WordPunctTokenizer
from nltk.corpus import stopwords
from torch import load, save
import torchtext


class Text2EmojiParser:
    """Class with function for process text and emoji data."""

    def __init__(self, pad_token: str, sos_token: str,
                 eos_token: str, unk_token: str):
        self.stop_words = set(stopwords.words("english"))
        self.tokenizer = WordPunctTokenizer()
        self.lemmatizer = WordNetLemmatizer()

        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self.emoji_vocab = None
        self.text_vocab = None
        self.max_length = None

    def tokenize_emoji(self, row: dict[str, str]) -> dict[str, list[str]]:
        """
        Tokenize emoji data.
        :param row: input dict[str, str] for processing
        :return: output dict[str, str] after processing
        """
        tokenized_row = list(map(lambda x: x if emoji.is_emoji(x) else "",
                                 row["emoji"]))
        tokenized_row = [self.sos_token] + tokenized_row + [self.eos_token]
        return {"tokenized_emoji": tokenized_row}

    def tokenize_text(self, row: dict[str, str], max_length: int) \
            -> dict[str, list[str]]:
        """
        Tokenize text data.
        :param row: input dict[str, str] for processing
        :param max_length: maximum length of text row
        :return: output dict[str, list[str]] after processing
        """
        self.max_length = max_length
        tokenized_row = list(
            filter(lambda x: x not in self.stop_words and x.isalpha(),
                   map(lambda x: self.lemmatizer.lemmatize(x),
                       nltk.word_tokenize(row["text"].lower())[:max_length])))
        tokenized_row = [self.sos_token] + tokenized_row + [self.eos_token]
        return {"tokenized_text": tokenized_row}

    def create_vocab(self, tokenized_emoji: Iterable, tokenized_text: Iterable,
                     min_freq_emoji: int = 0, min_freq_text: int = 0) -> None:
        """
        Create torchtext vocab from tokenized_emoji and tokenized_text rows.
        :param tokenized_emoji: tokenized emoji rows
        :param tokenized_text: tokenized text rows
        :param min_freq_emoji: minimum frequency of emoji in tokenized_emoji
        :param min_freq_text: minimum frequency of word in tokenized_text
        :return: None
        """
        specials = [self.pad_token, self.sos_token,
                    self.eos_token, self.unk_token]
        self.emoji_vocab = torchtext.vocab.build_vocab_from_iterator(
            tokenized_emoji,
            min_freq=min_freq_emoji,
            specials=specials)
        self.text_vocab = torchtext.vocab.build_vocab_from_iterator(
            tokenized_text,
            min_freq=min_freq_text,
            specials=specials)

    def set_default_index(self, idx: int) -> None:
        """
        Set default index for torchtext vocab.
        :param idx: default index
        :return: None
        """
        self.emoji_vocab.set_default_index(idx)
        self.text_vocab.set_default_index(idx)

    def numericalize_data(self, row: dict[str, list[str]]) \
            -> dict[str, list[int]]:
        """
        Set indexes instead of tokens.
        :param row: input dict[str, list[str]] for processing
        :return: output dict[str, list[str]] after processing
        """
        assert self.emoji_vocab is not None and self.text_vocab is not None

        emoji_ids = self.emoji_vocab.lookup_indices(row["tokenized_emoji"])
        text_ids = self.text_vocab.lookup_indices(row["tokenized_text"])
        return {"emoji_ids": emoji_ids, "text_ids": text_ids}

    def text_vocab_size(self) -> int:
        """
        Return size of text vocab.
        :return: size of text vocab
        """
        return len(self.text_vocab)

    def emoji_vocab_size(self) -> int:
        """
        Return size of emoji vocab.
        :return: Return size of emoji vocab.
        """
        return len(self.emoji_vocab)

    def save(self, path: str) -> None:
        """
        Save text and emoji vocabs.
        :param path: path for save
        :return: None
        """
        save((self.text_vocab, self.emoji_vocab, self.max_length), path)

    def load(self, path: str) -> None:
        """
        Load text and emoji vocabs.
        :param path: path for load
        :return: None
        """
        self.text_vocab, self.emoji_vocab, self.max_length = load(path)

    def tokenize(self, row: dict[str, str]) -> dict[str, list[int]]:
        """
        Tokenize row of text.
        :param row: input dict[str, str] for processing
        :return: output dict[str, list[str]] after processing
        """
        assert self.emoji_vocab is not None and \
               self.text_vocab is not None and \
               self.max_length is not None

        return self.text_vocab.lookup_indices(
            self.tokenize_text(row, self.max_length)["tokenized_text"])
