import torch
import torch.nn as nn
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(predictions, labels, save_path=None):
    cm = ConfusionMatrixDisplay.from_predictions(labels, predictions, normalize='true')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
# Train and Test functions
def train_one_epoch(loader, model, device, optimizer, criterion, log_interval, epoch):
    model.train()
    total_loss = 0
    correct = 0
    losses = []
    counter = []

    for i, (img, label) in enumerate(loader):
        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
        
        if (i + 1) % log_interval == 0:
            losses.append(loss.item())
            counter.append((i * loader.batch_size) + len(img) + epoch * len(loader.dataset))

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy, losses, counter

def evaluate(loader, model, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            outputs = model(img)
            loss = criterion(outputs, label)
            total_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=False)
            correct += pred.eq(label).sum().item()

            predictions.extend(pred.cpu().numpy())
            true_labels.extend(label.cpu().numpy())
            outputs.extend(output.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy, predictions, true_labels, outputs


def plot(log_counter, log_losses, val_losses, train_losses, val_accuracies, train_accuracies, save_path=None):

    plt.figure(figsize=(12, 6))
    plt.plot(log_counter, log_losses, color='blue', label='Training Loss')
    plt.legend()
    plt.title("Training Loss"); plt.ylabel('Loss')
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(val_losses)), val_losses, color='orange', label='Validation Loss')
    plt.plot(range(len(train_losses)), train_losses, color='green', label='Training Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.title("Validation & Training Loss Over Epochs")
    plt.legend()
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(val_accuracies)), val_accuracies, color='orange', label='Validation Accuracy')
    plt.plot(range(len(train_accuracies)), train_accuracies, color='green', label='Training Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy')
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
