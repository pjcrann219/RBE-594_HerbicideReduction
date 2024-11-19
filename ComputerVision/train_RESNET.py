from Utilities import *
from models import *
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import time
import os

def get_RESNET():

    orig_weights = ResNet50_Weights.IMAGENET1K_V1
    resnet = resnet50(weights=orig_weights)

    original_conv = resnet.conv1
    resnet.conv1 = nn.Conv2d(
        in_channels=4, 
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias
    )

    # Copy weights for the first 3 channels and initialize the 4th channel
    with torch.no_grad():
        resnet.conv1.weight[:, :3, :, :] = original_conv.weight  # Copy weights for RGB
        resnet.conv1.weight[:, 3:, :, :] = torch.mean(original_conv.weight, dim=1, keepdim=True)  # Initialize 4th channel

    # Change number of output layers -> TODO: can try multiple layers in classification head
    resnet.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=1),
        nn.Sigmoid()  # Sigmoid activation
    )

    return resnet

print(f"Starting experiment")
mlflow.set_experiment('RESNET')

hyper_params = {
    "num_epochs": 20,
    "learning_rate": 0.0001,
    "weight_decay": 1e-4,
    "batch_size": 16,
    "num_workers": 0,
    "shuffle": True,
    "threshold": 0.5,
    "num_images": 5000,
    "balance_ratio": 0.50
}
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
    num_images = 1000,#hyper_params['num_images']
    balance_ratio=hyper_params['balance_ratio']
)

# check for cuda or mps on mac, if not, use cpu
device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
print("Using device: ", device)

# model = BinaryCNN().to(device)
# model = CNN_512_4().to(device)
model = get_RESNET()
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
        optimizer.zero_grad() # Zero gradiants

        # Send inputs and labels to device
        input = input.to(device) 
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(input)
        outputs = outputs.flatten(0)
        predictions = (outputs>= hyper_params["threshold"]).float()

        # Compute loss/accuracy and do back prop
        loss = criterion(outputs, labels)
        loss.backward()
        accuracy = sum(predictions == labels) / labels.size(0)

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
            outputs = outputs.flatten(0)
            predictions = (outputs>= hyper_params["threshold"]).float()

            # Compute loss/accuracy
            loss = criterion(outputs, labels)
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

torch.save(model, 'models/RESNET/full_model1.pth')
torch.save(model.state_dict(), "models/RESNET/state_dict1.pth")

# mlflow.pytorch.log_model(model, "model", input_example=input.cpu().numpy())
# mlflow.pytorch.log_model(
#     model, 
#     "model",
#     input_example=input.cpu().numpy().astype('float32')
# )
mlflow.end_run()