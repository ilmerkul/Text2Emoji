from IPython.display import clear_output
import matplotlib.pyplot as plt

import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.optim import Adam, lr_scheduler

import tqdm
from datetime import date

import evaluate

bleu = evaluate.load('bleu')


def evaluate_bleu(model, dataset, device):
    train_dataset = dataset.get_test()
    references = list(map(lambda x: [' '.join(map(str, x.tolist()[1:-1]))], train_dataset['emoji_ids']))
    predictions = []
    for k in range(len(train_dataset['text_ids'])):
        source = train_dataset['text_ids'][k].unsqueeze(-1)
        source = source.to(device=device)
        prediction = model.translate()
        predictions.append(' '.join(map(str, prediction[0][:-1].tolist())))
    results = bleu.compute(predictions=predictions, references=references, tokenizer=str.split, max_order=2)

    return results


def evaluate_loss_test(model, test_data_loader, loss, emoji_vocab_size, device):
    mean_loss = 0
    model.eval()

    with torch.no_grad():
        for batch in test_data_loader:
            batch_en_ids = batch['en_ids']
            batch_de_ids = batch['de_ids']
            batch_en_ids = batch_en_ids.to(device=device)
            batch_de_ids = batch_de_ids.to(device=device)

            logits = model(batch_en_ids, batch_de_ids)
            loss_t = loss(logits, one_hot(batch_de_ids.permute(1, 0)[:, 1:],
                                          num_classes=emoji_vocab_size).to(torch.float))

            mean_loss += loss_t.item()

    return mean_loss / len(test_data_loader)


def print_learn_curve(history):
    # clear_output(True)
    plt.close('all')
    plt.figure(figsize=(12, 4))
    for i, (name, h) in enumerate(sorted(history.items())):
        plt.subplot(1, len(history), i + 1)
        plt.title(name)
        plt.plot(range(len(h)), h)
        plt.grid()
    plt.savefig(f'./data/learning_curves/curve_{date.today()}')
    # plt.show()


def train_model(model, dataset, n_epoch, print_step, emoji_vocab_size, pad_idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_milestones = [2, 4, 7]
    batch_sizes = [32, 64, 128, 256]
    epoch_emb_requires_grad = 4

    model.to(device=device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 7], gamma=0.464159)
    loss = CrossEntropyLoss()

    history = {'train_loss': [], 'test_loss': []}
    batch_step = 0
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
            batch_en_ids = batch_en_ids.to(device=device)
            batch_de_ids = batch_de_ids.to(device=device)

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
                mean_test_loss = evaluate_loss_test(model, test_data_loader, loss, emoji_vocab_size, device)
                print(f'step: {i}/{len(train_data_loader)}, '
                      f'train_loss: {mean_train_loss}, '
                      f'test_loss: {mean_test_loss}')
                history['train_loss'].append(mean_train_loss)
                history['test_loss'].append(mean_test_loss)

                # save state
                torch.save({
                    'epoch': epoch,
                    'history': history,
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'batch_size': batch_size,
                    'loss': loss
                }, f'./data/checkpoints/checkpoint_{date.today()}.pth')

                # plot learning curve
                print_learn_curve(history)

                # callbacks
                if len(history['test_loss']) > 1 and history['test_loss'][-2] < history['test_loss'][-1]:
                    test_learn_curve_increases += 1
                else:
                    test_learn_curve_increases = 0

                if test_learn_curve_increases > 5:
                    return history

        # calculate bleu
        results = evaluate_bleu(model, dataset, device)
        print(f'bleu: {results}')

        scheduler.step()
    model.eval()
    model.to('cpu')

    return history
