import matplotlib.pyplot as plt

import torch
from torch.nn.functional import one_hot

from datetime import date

import evaluate

bleu = evaluate.load('bleu')


def load_checkpoint(path, model, optim, scheduler, loss):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    loss = torch['loss']
    optim.load_state_dict(checkpoint['optim'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    history = checkpoint['history']
    batch_size = checkpoint['batch_size']
    epoch = checkpoint['epoch']

    return {'model': model, 'loss': loss, 'optim': optim,
            'scheduler': scheduler, 'history': history,
            'batch_size': batch_size, 'epoch': epoch}


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
    plt.close('all')
    plt.figure(figsize=(12, 4))
    for i, (name, h) in enumerate(sorted(history.items())):
        plt.subplot(1, len(history), i + 1)
        plt.title(name)
        plt.plot(range(len(h)), h)
        plt.grid()
    plt.savefig(f'./data/learning_curves/curve_{date.today()}')
