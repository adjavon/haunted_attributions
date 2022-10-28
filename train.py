"""Training a Vgg Network on the Haunted dataset."""
from pathlib import Path

import torch
from funlib.learn.torch.models import Vgg2D
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import get_haunted_dataset


def cycle(iterable):
    """Infinitely run through a finite-size iterable."""
    while True:
        for x in iterable:
            yield x


@torch.no_grad()
def evaluate(model, dataloader, device):
    preds = []
    gt = []
    for x, y in tqdm(dataloader, total=len(dataloader),
                     desc="Validate"):
        logits = model(x.to(device))
        pred = torch.argmax(logits, dim=1)
        preds.append(pred.cpu())
        gt.append(y)
    preds = torch.cat(preds)
    gt = torch.cat(gt)
    # accuracy
    return accuracy_score(gt.numpy(), preds.numpy())


if __name__ == "__main__":
    # General setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = device == "cuda"
    iterations = 1000
    batch_size = 16
    validate_every = 250
    save_dir = Path("models/")
    tboard = SummaryWriter()
    # Create the dataset and dataloader
    haunted = get_haunted_dataset()
    dataloader = DataLoader(haunted, batch_size=batch_size, shuffle=True,
                            drop_last=True, pin_memory=pin_memory)
    data_iter = cycle(dataloader)
    # Validation
    val_dataset = get_haunted_dataset(split="valid")
    val_dl = DataLoader(haunted, batch_size=batch_size*2, shuffle=False,
                        drop_last=False, pin_memory=pin_memory)
    # Create the model, loss, optimizer
    model = Vgg2D(input_size=(256, 256),
                  fmaps=8,
                  output_classes=2,
                  input_fmaps=3)
    model.to(device)
    bce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Loop
    val_accuracy = 0
    for i in tqdm(range(iterations)):
        x, y = next(data_iter)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = bce(pred, y)
        loss.backward()
        optimizer.step()
        if i > 0 and i % validate_every == 0:
            # validate
            accuracy = evaluate(model, dataloader, device)
            print(accuracy)
            # Add to tensorboard
            tboard.add_scalar("train/loss", loss.item(),
                              global_step=i)
            tboard.add_scalar("validation/accuracy", accuracy,
                              global_step=i)
            if accuracy > val_accuracy:
                # to save
                val_accuracy = accuracy
                torch.save(model.state_dict(), save_dir / "model.pth")
