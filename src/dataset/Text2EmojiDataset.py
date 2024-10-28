from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import cpu


class Text2EmojiDataset(Dataset):
    def __init__(self):
        self.dataset = None

    def download_and_tokenization_dataset(self, tokenize_emoji_func, tokenize_text_func, max_text_length, seed=42):
        self.dataset = load_dataset('KomeijiForce/Text2Emoji', split='train')
        self.dataset.shuffle(seed=seed)

        def check_none(row):
            return row['text'] is not None and row['emoji'] is not None and row['topic'] is not None

        self.dataset = self.dataset.filter(check_none)

        self.dataset = self.dataset.map(tokenize_emoji_func, num_proc=cpu.device_count())
        self.dataset = self.dataset.map(tokenize_text_func, fn_kwargs={'max_length': max_text_length},
                                        num_proc=cpu.device_count())

    def numericalize_dataset(self, numericalize_func):
        self.dataset = self.dataset.map(numericalize_func)

    def get_collate_fn(self, pad_index):
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

    def train_test_split(self, train_test_ratio):
        data_type = "torch"
        format_columns = ["emoji_ids", "text_ids"]
        self.dataset = self.dataset.with_format(type=data_type, columns=format_columns, output_all_columns=True)

        self.dataset = self.dataset.train_test_split(test_size=train_test_ratio)

    def get_data_loader(self, batch_size, pad_idx):
        collate_fn = self.get_collate_fn(pad_idx)
        train_data_loader = DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
        )
        test_data_loader = DataLoader(
            dataset=self.dataset['test'],
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )

        return train_data_loader, test_data_loader

    def save(self, path):
        self.dataset.save_to_disk(path)

    def load(self, path):
        self.dataset = load_from_disk(path)

    def get_test(self):
        return self.dataset['test']

    def get_train(self):
        return self.dataset['train']

    def __len__(self):
        assert self.dataset is not None, 'download dataset'
        return len(self.dataset)

    def __getitem__(self, item):
        assert self.dataset is not None, 'download dataset'
        return self.dataset[item]
