import emoji
from nltk.corpus import stopwords
import nltk
from nltk import WordPunctTokenizer, WordNetLemmatizer

import torchtext

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class Text2EmojiParser:
    def __init__(self, pad_token, sos_token, eos_token, unk_token):
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = WordPunctTokenizer()
        self.lemmatizer = WordNetLemmatizer()

        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self.emoji_vocab = None
        self.text_vocab = None
        self.max_length = None

    def tokenize_emoji(self, row):
        tokenized_row = list(map(lambda x: x if emoji.is_emoji(x) else '', row['emoji']))
        tokenized_row = [self.sos_token] + tokenized_row + [self.eos_token]
        return {'tokenized_emoji': tokenized_row}

    def tokenize_text(self, row, max_length):
        self.max_length = max_length
        tokenized_row = list(filter(lambda x: x not in self.stop_words and x.isalpha(),
                                    map(lambda x: self.lemmatizer.lemmatize(x),
                                        nltk.word_tokenize(row['text'].lower())[:max_length])))
        tokenized_row = [self.sos_token] + tokenized_row + [self.eos_token]
        return {'tokenized_text': tokenized_row}

    def create_vocab(self, tokenized_emoji, tokenized_text, min_freq_emoji=0, min_freq_text=0):
        specials = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        self.emoji_vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_emoji,
                                                                     min_freq=min_freq_emoji,
                                                                     specials=specials)
        self.text_vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_text,
                                                                    min_freq=min_freq_text,
                                                                    specials=specials)

    def set_default_index(self, idx):
        self.emoji_vocab.set_default_index(idx)
        self.text_vocab.set_default_index(idx)

    def numericalize_data(self, row):
        assert self.emoji_vocab is not None and self.text_vocab is not None

        emoji_ids = self.emoji_vocab.lookup_indices(row['tokenized_emoji'])
        text_ids = self.text_vocab.lookup_indices(row['tokenized_text'])
        return {'emoji_ids': emoji_ids, 'text_ids': text_ids}

    def text_vocab_size(self):
        return len(self.text_vocab)

    def emoji_vocab_size(self):
        return len(self.emoji_vocab)

    def tokenize(self, row):
        assert self.emoji_vocab is not None and self.text_vocab is not None and self.max_length is not None

        return self.text_vocab.lookup_indices(self.tokenize_text(row, self.max_length)['tokenized_text'])
