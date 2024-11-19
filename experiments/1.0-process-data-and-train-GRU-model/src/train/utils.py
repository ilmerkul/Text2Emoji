from datetime import date

import evaluate
import matplotlib.pyplot as plt
import torch
from src.dataset import Text2EmojiDataset
from src.model import Text2Emoji
from torch.nn.functional import one_hot

bleu = evaluate.load("bleu")


def load_checkpoint(path: str, model: Text2Emoji, optim: torch.optim.Adam,
                    scheduler: torch.optim.lr_scheduler) -> dict:
    """
    Load state of model's learning from path.
    :param path: path for load checkpoint
    :param model: torch model for load state checkpoint['model']
    :param optim: optimizer for load state checkpoint['optim']
    :param scheduler: lr_scheduler for load state checkpoint['scheduler']
    :return: dict - {'model': model, 'loss': loss, 'optim': optim,
                     'scheduler': scheduler, 'history': history,
                     'batch_size': batch_size, 'epoch': epoch}
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    loss = torch["loss"]
    optim.load_state_dict(checkpoint["optim"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    history = checkpoint["history"]
    batch_size = checkpoint["batch_size"]
    epoch = checkpoint["epoch"]

    return {"model": model, "loss": loss, "optim": optim,
            "scheduler": scheduler, "history": history,
            "batch_size": batch_size, "epoch": epoch}


def evaluate_bleu(model: Text2Emoji, dataset: Text2EmojiDataset,
                  device: torch.device) -> float:
    """
    Evaluate bleu metric of torch model on dataset.
    :param model: torch model
    :param dataset: class Text2EmojiDataset dataset
    :param device: torch device (CPU/GPU/TPU)
    :return: value of bleu metric on test data
    """
    train_dataset = dataset.get_test()
    references = list(map(lambda x: [" ".join(map(str, x.tolist()[1:-1]))],
                          train_dataset["emoji_ids"]))
    predictions = []
    model.to(device)
    for k in range(len(train_dataset["text_ids"])):
        source = train_dataset["text_ids"][k].unsqueeze(-1)
        source = source.to(device=device)
        prediction = model.translate(source)
        predictions.append(" ".join(map(str, prediction[0][:-1].tolist())))
    results = bleu.compute(predictions=predictions, references=references,
                           tokenizer=str.split, max_order=2)

    return results


def evaluate_loss_test(model: Text2Emoji,
                       test_data_loader: torch.utils.data.DataLoader,
                       loss: torch.nn.CrossEntropyLoss, num_classes: int,
                       device: torch.device) -> float:
    """
    Evaluate loss of torch model on test_data_loader.
    :param model: torch model
    :param test_data_loader: torch Dataloader
    :param loss: torch cross entropy loss function
    :param num_classes: count of classes
    :param device: torch device (CPU/GPU/TPU)
    :return: average value of loss on test data
    """
    mean_loss = 0
    model.eval()

    with torch.no_grad():
        for batch in test_data_loader:
            batch_en_ids = batch["en_ids"]
            batch_de_ids = batch["de_ids"]
            batch_en_ids = batch_en_ids.to(device=device)
            batch_de_ids = batch_de_ids.to(device=device)

            logits = model(batch_en_ids, batch_de_ids)
            loss_t = loss(logits, one_hot(batch_de_ids.permute(1, 0)[:, 1:],
                                          num_classes=num_classes).to(
                torch.float))

            mean_loss += loss_t.item()
    mean_loss = mean_loss / len(test_data_loader)
    return mean_loss


def save_learn_curve(history: dict[str, list[float]], path: str) -> None:
    """
    Plot and save learn curve.
    :param history: Dict with train and test loss history.
    :param path: path for save curve
    :return: None
    """
    plt.close("all")
    plt.figure(figsize=(12, 4))
    for i, (name, h) in enumerate(sorted(history.items())):
        plt.subplot(1, len(history), i + 1)
        plt.title(name)
        plt.plot(range(len(h)), h)
        plt.grid()
    plt.savefig(f"{path}/curve_{date.today()}")
