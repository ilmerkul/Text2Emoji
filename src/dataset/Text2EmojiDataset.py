from typing import Callable, Iterable

import datasets
import torch
from torch import cpu
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class Text2EmojiDataset(Dataset):
    """Class of dataset for Text2Emoji training."""

    def __init__(self, dataset: datasets.Dataset):
        self.dataset = dataset

    def shuffle(self, seed: int = 42) -> None:
        """
        Shuffle data.
        :param seed: random seed
        :return: None
        """
        self.dataset.shuffle(seed=seed)

    def filter_none(self) -> None:
        """
        Drop none rows.
        :return: None
        """

        def check_none(row: dict[str, str]) -> bool:
            return row["text"] is not None and \
                   row["emoji"] is not None and \
                   row["topic"] is not None

        self.dataset = self.dataset.filter(check_none)

    def tokenization_dataset(self, tokenize_emoji_func: Callable,
                             tokenize_text_func: Callable,
                             max_text_length: Callable) -> None:
        """
        Tokenize dataset with help of tokenize_emoji_func and
         tokenize_text_func function.
        :param tokenize_emoji_func: function for emoji tokenize
        :param tokenize_text_func: function for text tokenize
        :param max_text_length: maximum length of text sentence
        :return: None
        """
        self.dataset = self.dataset.map(tokenize_emoji_func,
                                        num_proc=cpu.device_count())
        self.dataset = self.dataset.map(tokenize_text_func, fn_kwargs={
            "max_length": max_text_length},
                                        num_proc=cpu.device_count())

    def numericalize_dataset(self, numericalize_func: Callable) -> None:
        """
        Numericalize dataset with help of numericalize_func function.
        :param numericalize_func: function for numericalize
        :return: None
        """
        self.dataset = self.dataset.map(numericalize_func)

    def get_collate_fn(self, pad_index: int) -> Callable:
        """
        Return collate function for create of Dataloader.
        :param pad_index: index of pad token
        :return: collate function
        """

        def collate_fn(batch: Iterable) -> dict[str, torch.Tensor]:
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

    def train_test_split(self, train_test_ratio: float) -> None:
        """
        Split on train and test dataset.
        :param train_test_ratio: train test ration
        :return: None
        """
        data_type = "torch"
        format_columns = ["emoji_ids", "text_ids"]
        self.dataset = self.dataset.with_format(type=data_type,
                                                columns=format_columns,
                                                output_all_columns=True)

        self.dataset = self.dataset.train_test_split(
            test_size=train_test_ratio)

    def get_data_loader(self, batch_size: int, pad_idx: int) -> (
            torch.utils.data.DataLoader, torch.utils.data.DataLoader):
        """
        Create train and test DataLoader.
        :param batch_size: size of batch
        :param pad_idx: index of pad token
        :return: tuple of train and test DataLoader
        """
        collate_fn = self.get_collate_fn(pad_idx)
        train_data_loader = DataLoader(
            dataset=self.dataset["train"],
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
        )
        test_data_loader = DataLoader(
            dataset=self.dataset["test"],
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )

        return train_data_loader, test_data_loader

    def save(self, path: str) -> None:
        """
        Save dataset on disk.
        :param path: path for save
        :return: None
        """
        self.dataset.save_to_disk(path)

    def get_test(self) -> dict[str, list[int]]:
        """
        Return test dataset.
        :return: test dataset
        """
        return self.dataset["test"]

    def get_train(self) -> dict[str, list[int]]:
        """
        Return train dataset.
        :return: train dataset
        """
        return self.dataset["train"]

    def __len__(self):
        assert self.dataset is not None, "download dataset"
        return len(self.dataset)

    def __getitem__(self, item):
        assert self.dataset is not None, "download dataset"
        return self.dataset[item]
