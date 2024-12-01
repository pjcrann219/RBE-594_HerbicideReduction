from Utilities import *
from models import *
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from datetime import datetime
import time
import os
from load_ViT import *
import transformers
transformers.logging.set_verbosity_error()

run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"Starting experiment")
mlflow.set_experiment('ViT')

hyper_params = {
    "num_epochs": 10,
    "learning_rate": 0.0001,
    "weight_decay": 1e-3,
    "batch_size": 16,
    "num_workers": 0,
    "shuffle": True,
    "threshold": 0.5,
    "num_images": 22000,
    "balance_ratio": 0.50
}

print(f"Starting run - {run_name}")
mlflow.start_run(run_name=run_name)
mlflow.log_params(hyper_params)

train_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Agriculture-Vision-2021/train")

print(f"Loading training dataloader... ", end='')
train_loader = get_dataloader(
    train_dir, 
    batch_size=hyper_params['batch_size'],
    num_workers=hyper_params['num_workers'],
    shuffle=hyper_params['shuffle'],
    num_images = hyper_params['num_images'],
    balance_ratio=hyper_params['balance_ratio']
)

# test_dir = "ComputerVision/Agriculture-Vision-2021/val"
test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Agriculture-Vision-2021/val")

print(f"Loading testing dataloader... ", end='')
test_loader = get_dataloader(
    test_dir,
    batch_size=hyper_params['batch_size'],
    num_workers=hyper_params['num_workers'],
    shuffle=hyper_params['shuffle'],
    num_images = 2000,#hyper_params['num_images']
    balance_ratio=hyper_params['balance_ratio']
)

# check for cuda or mps on mac, if not, use cpu
device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
print("Using device: ", device)

model = get_ViT()
model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=hyper_params['learning_rate'], weight_decay=hyper_params['weight_decay'])

data = TrainingData()

for epoch in range(hyper_params['num_epochs']):
    epoch_start_time = time.time()

    # Training
    model.train()
    epoch_loss_train = 0
    epoch_accuracy_train = 0
    for input, labels, in train_loader:
        # print(f"input: {input.shape}, label: {labels.shape}")
        optimizer.zero_grad() # Zero gradiants

        # Send inputs and labels to device
        input = input.to(device) 
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(input)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits.flatten())
        predictions = (probabilities >= hyper_params["threshold"]).float()

        # Compute loss/accuracy and do back prop
        loss = criterion(probabilities, labels)
        loss.backward()
        accuracy = (predictions == labels).float().mean().item()

        # Step optimizer
        optimizer.step()

        # Sum epoch details
        epoch_loss_train += loss.item()
        epoch_accuracy_train += accuracy

    epoch_loss_train = epoch_loss_train / len(train_loader)
    epoch_accuracy_train = epoch_accuracy_train / len(train_loader)


    # Testing
    model.eval()
    epoch_loss_test = 0
    epoch_accuracy_test = 0
    with torch.no_grad():
        for input, labels, in test_loader:

            # Send inputs and labels to device
            input = input.to(device) 
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(input)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits.flatten())
            predictions = (probabilities >= hyper_params["threshold"]).float()

            # Compute loss/accuracy
            loss = criterion(predictions, labels)
            accuracy = sum(predictions == labels).item() / labels.size(0)

            # Sum epoch details
            epoch_loss_test += loss.item()
            epoch_accuracy_test += accuracy

    epoch_loss_test = epoch_loss_test / len(test_loader)
    epoch_accuracy_test = epoch_accuracy_test / len(test_loader)

    elapsed = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{hyper_params['num_epochs']} - Loss: {epoch_loss_train:.4f} - Accuracy: {epoch_accuracy_train:.4f} - Test Loss: {epoch_loss_test:.4f} - Test Accuracy: {epoch_accuracy_test:.4f} - Time: {elapsed:.2f}s")
    data.training_loss.append(epoch_loss_train)
    data.training_accuracy.append(epoch_accuracy_train)
    # print(epoch_loss / len(train_loader))

    mlflow.log_metrics({
        "train_loss": epoch_loss_train,
        "train_accuracy": epoch_accuracy_train,
        "test_loss": epoch_loss_test,
        "test_accuracy": epoch_accuracy_test
    }, step=epoch)

model = model.cpu()

save_model(model, 'ViT', run_name)

mlflow.end_run()