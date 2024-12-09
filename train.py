from utils import train_one_epoch, evaluate, plot
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import trange

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Transforms
data_transforms = {
    "train": transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Keep greyscale
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    "val": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    "test": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
}


# Load Dataset
def load_data(data_dir):
    datasets = {x: ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x])
                for x in ["train", "val", "test"]}
    dataloaders = {x: DataLoader(datasets[x], batch_size=16, shuffle=True, num_workers=4)
                   for x in ["train", "val", "test"]}
    return dataloaders

def main():
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument("--model", type=str, required=True, choices=["simple_cnn", "vgg16_untrained","vgg16_default", "autoencoder", "resnet"], help="Model type")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=2, help="Patience for Validation")
    parser.add_argument("--seed", type=int, default=567, help="Set seed")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory to save plots and metrics")
    args = parser.parse_args()

    # reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)

    # Load data
    dataloaders = load_data(args.data_dir)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # Instantiate model
    model_module = importlib.import_module(f"models.{args.model}")
    if args.model == "simple_cnn":
        model = model_module.SimpleCNN()
    elif args.model == "vgg16_untrained":
        model = model_module.VGG16()
    elif args.model == "vgg16_default":
        model = model_module.VGG16()
    elif args.model == "resnet":
        model = model_module.ResNet18()
    else:
        model = model_module.AE()    
    model = model.to(DEVICE)

    # optimizer and criterion
    optimizer = optim.Adagrad(model.parameters(), lr=0.00001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # track metrics
    metrics_list = []

    # train loop
    best_val_loss = float("inf")
    patience = args.patience
    p_counter = 0
    log_losses = []
    log_counter = []
    
    for epoch in trange(args.epochs, desc="Epochs"):
        train_loss, train_accuracy, train_log_losses, train_log_counter = train_one_epoch(
            train_loader, model, DEVICE, optimizer, criterion, 7, epoch
        )
        log_losses.extend(train_log_losses)  # log losses
        log_counter.extend(train_log_counter)  # log counters
        
        val_loss, val_accuracy, _, _, _ = evaluate(val_loader, model, DEVICE, criterion)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")

        metrics_list.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        })

        # validation stopage
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.save_dir}/best_model.pth")
            p_counter = 0
        else:
            p_counter += 1
            if p_counter >= patience:
                print("Early stopping triggered!")
                break

    # do 
    model.load_state_dict(torch.load(f"{args.save_dir}/best_model.pth"))
    test_loss, test_accuracy, predictions, labels, outputs = evaluate(test_loader, model, DEVICE, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save metrics to a pandas dataframe and CSV
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df["test_loss"] = test_loss  # Add test loss as a separate column
    metrics_df["test_accuracy"] = test_accuracy  # Add test accuracy as a separate column
    metrics_df.to_csv(f"{args.save_dir}/metrics.csv", index=False)

    # Save predictions and labels for further analysis
    probabilities = torch.softmax(torch.tensor(outputs), dim=1).numpy()
    predictions_df = pd.DataFrame({
    "predictions": predictions,
    "true_labels": labels,
    "probabilities_class_0": probabilities[:, 0],
    "probabilities_class_1": probabilities[:, 1]
    })
    predictions_df.to_csv(f"{args.save_dir}/test_predictions.csv", index=False)

    # Plot training and validation metrics
    plot(
        log_counter, log_losses,
        val_losses=[m["val_loss"] for m in metrics_list],
        train_losses=[m["train_loss"] for m in metrics_list],
        val_accuracies=[m["val_accuracy"] for m in metrics_list],
        train_accuracies=[m["train_accuracy"] for m in metrics_list],
        save_path=f"{args.save_dir}/plots.png"
    )
    
    plot_confusion_matrix(predictions, labels, save_path=f"{args.save_dir}/confusion_matrix.png")


if __name__ == "__main__":
    main()
