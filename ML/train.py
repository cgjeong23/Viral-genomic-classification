import os

from tqdm.notebook import tqdm
import torch
from sklearn.metrics import classification_report, accuracy_score


def train(model,
          dataloader,
          loss_function,
          lr,
          num_epochs,
          valid_loader=None,
          test_loader=None):
    os.makedirs('/content/drive/MyDrive/GeneModels', exist_ok=True)
    # pytorch training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train': [], 'valid': [], 'test': []}
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader)

        all_preds = []
        all_labels = []
        for batch in pbar:

            batch_sequences, y = batch
            x = batch_sequences.to('cuda')
            y = y.to('cuda')

            h = model(x)  # [B, C]
            j = loss_function(h, y)

            # do gradient descent
            optimizer.zero_grad()  # remove junk from last step
            j.backward()  # calculate gradient from current batch outputs
            optimizer.step()  # update the weights using the gradients

            all_preds.append(h.argmax(-1).detach().cpu())
            all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        print(classification_report(all_labels, all_preds, digits=4))
        accuracy = accuracy_score(all_labels, all_preds)

        if valid_loader is not None:
            val_accuracy = evaluate(model, valid_loader)
            history['valid'].append(val_accuracy)
        if test_loader is not None:
            test_accuracy = evaluate(model, test_loader)
            history['test'].append(test_accuracy)

        history['train'].append(accuracy)

        torch.save(model.state_dict(), f'/content/drive/MyDrive/GeneModels/{epoch}.pth')

    return history


def evaluate(model, valid_loader):
    valid_preds = []
    valid_labels = []
    for batch in valid_loader:
        batch_sequences, y = batch
        x = batch_sequences.to('cuda')
        y = y.to('cuda')
        h = model(x)  # [B, C]
        valid_preds.append(h.argmax(-1).detach().cpu())
        valid_labels.append(y.cpu())
    valid_preds = torch.cat(valid_preds).numpy()
    valid_labels = torch.cat(valid_labels).numpy()

    print(classification_report(valid_labels, valid_preds, digits=4))
    accuracy = accuracy_score(valid_labels, valid_preds)
    return accuracy