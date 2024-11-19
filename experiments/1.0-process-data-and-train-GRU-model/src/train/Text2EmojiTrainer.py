from datetime import date
from typing import Iterable

import mlflow
import torch
import tqdm
from datasets import Dataset
from loguru import logger
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.optim import Adam, lr_scheduler

from .utils import evaluate_bleu, evaluate_loss_test


class Text2EmojiTrainer(object):
    """Class for train Text2Emoji model."""

    def __init__(self, model: torch.nn.Module, train_cfg: DictConfig):
        self.train_cfg = train_cfg
        self.model = model

        self.optimizer = self.get_optimizer(train_cfg.lr_0)
        self.scheduler = self.get_scheduler(self.optimizer,
                                            train_cfg.lr_milestones,
                                            train_cfg.gamma)
        self.loss = self.get_loss()

    def get_loss(self) -> torch.nn.modules.loss.CrossEntropyLoss:
        """
        Return loss function.
        :return: loss function
        """
        loss = CrossEntropyLoss()
        return loss

    def get_scheduler(self, optimizer: torch.optim.adam.Adam,
                      lr_milestones: Iterable[int],
                      gamma: float) -> torch.optim.lr_scheduler.MultiStepLR:
        """
        Return scheduler.
        :param optimizer: optimizer
        :param lr_milestones: milestones for milestones
        :param gamma: gamma for MultiStepLR
        :return: scheduler
        """
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=lr_milestones,
                                             gamma=gamma)
        return scheduler

    def get_optimizer(self, lr_0: float) -> torch.optim.adam.Adam:
        """
        Return optimizer.
        :param lr_0: lr for Adam
        :return: optimizer
        """
        optimizer = Adam(self.model.parameters(), lr=lr_0)
        return optimizer

    def train(self, dataset: Dataset, path_save: str):
        """
        Train model.
        :param dataset: hf dataset
        :param path_save: path save model
        :return: history loss
        """
        device = self.model.enc.device
        emoji_vocab_size = self.model.de_vocab_size
        pad_idx = self.model.pad_id

        n_epoch = self.train_cfg.epoch
        print_step = self.train_cfg.print_step
        batch_milestones = self.train_cfg.batch_milestones
        batch_sizes = self.train_cfg.batch_sizes
        epoch_emb_requires_grad = self.train_cfg.epoch_emb_requires_grad

        params = {
            "epochs": self.train_cfg.epoch,
            "batch_milestones": self.train_cfg.batch_milestones,
            "batch_sizes": self.train_cfg.batch_sizes,
            "epoch_emb_requires_grad": self.train_cfg.epoch_emb_requires_grad,
            "start_learning_rate": self.train_cfg.lr_0,
            "optimizer": "Adam",
            "lr_scheduler": {
                "type": "MultiStepLR",
                "milestones": self.train_cfg.lr_milestones,
                "gamma": self.train_cfg.gamma,
            },
            "loss_function": self.loss.__class__.__name__,
            "metric_function": 'bleu',
        }

        mlflow.log_params(params)

        history = {'train_loss': [], 'test_loss': []}
        batch_step = 0
        train_loss = 0

        test_learn_curve_increases = 0

        train_data_loader, test_data_loader = dataset.get_data_loader(
            batch_sizes[0], pad_idx)
        for epoch in range(n_epoch):
            if epoch == epoch_emb_requires_grad:
                self.model.emb_requires_grad()

            if epoch in batch_milestones:
                train_data_loader, test_data_loader = dataset.get_data_loader(
                    batch_sizes[batch_step], pad_idx)
                batch_step += 1
            batch_size = batch_sizes[batch_step]

            logger.info(f'epoch: {epoch + 1}/{n_epoch}, '
                        f'lr: {self.scheduler.get_last_lr()}, '
                        f'batch_size: {batch_size}')
            for i, batch in enumerate(tqdm.tqdm(train_data_loader)):
                self.model.train()

                batch_en_ids = batch['en_ids']
                batch_de_ids = batch['de_ids']
                batch_en_ids = batch_en_ids.to(device=device)
                batch_de_ids = batch_de_ids.to(device=device)

                self.optimizer.zero_grad()

                logits = self.model(batch_en_ids, batch_de_ids)
                loss_t = self.loss(logits,
                                   one_hot(batch_de_ids.permute(1, 0)[:, 1:],
                                           num_classes=emoji_vocab_size).to(
                                       torch.float))
                loss_t.backward()
                self.optimizer.step()

                train_loss += loss_t.item()
                if i % print_step == 0 and i != 0:
                    self.model.eval()

                    # evaluate
                    mean_train_loss = train_loss / print_step
                    train_loss = 0
                    mean_test_loss = evaluate_loss_test(self.model,
                                                        test_data_loader,
                                                        self.loss,
                                                        emoji_vocab_size,
                                                        device)
                    mlflow.log_metric('train_loss', mean_train_loss,
                                      step=(i // print_step))
                    mlflow.log_metric('test_loss', mean_test_loss,
                                      step=(i // print_step))
                    logger.info(f'step: {i}/{len(train_data_loader)}, '
                                f'train_loss: {mean_train_loss}, '
                                f'test_loss: {mean_test_loss}')
                    history['train_loss'].append(mean_train_loss)
                    history['test_loss'].append(mean_test_loss)

                    # save state
                    checkpoint_path = f'{path_save}/checkpoint_{date.today()}.pth'
                    torch.save({
                        'epoch': epoch,
                        'history': history,
                        'model': self.model.state_dict(),
                        'optim': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'batch_size': batch_size,
                        'loss': self.loss
                    }, checkpoint_path)
                    mlflow.log_artifact(checkpoint_path)

                    # callbacks
                    if len(history['test_loss']) > 1 and history['test_loss'][
                        -2] < \
                            history['test_loss'][-1]:
                        test_learn_curve_increases += 1
                    else:
                        test_learn_curve_increases = 0

                    if test_learn_curve_increases > 5:
                        return history

            # calculate bleu
            results = evaluate_bleu(self.model, dataset, device)
            mlflow.log_metric('bleu', results, step=epoch)
            logger.info(f'bleu: {results}')

            self.scheduler.step()

        self.model.eval()
        return history
