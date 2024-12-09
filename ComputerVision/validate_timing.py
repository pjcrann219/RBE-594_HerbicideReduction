import torch
from Utilities import *
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from matplotlib import patches
import seaborn as sns
from time import time

# Evaluates model performance on test data and returns results as a DataFrame
def run_timing_eval(model, num_images=32, balance_ratio=0.50, isViT=False):

    all_data = []

    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Agriculture-Vision-2021/val")
    test_loader = get_dataloader(
        test_dir,
        batch_size=32,
        num_workers=0,
        shuffle=False,
        num_images=num_images,
        balance_ratio=balance_ratio
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
    # device = "cpu"

    batch_times = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            start_time = time()
            outputs = model(inputs.to(device))
            if isViT:
                outputs = torch.sigmoid(outputs.logits.flatten())
            outputs = outputs.cpu().numpy() if torch.is_tensor(outputs) else outputs
            labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
            all_data.extend([
                {'label': label, 'output': output}
                for label, output in zip(labels, outputs)
            ])
            end_time = time()

            batch_times.append(end_time - start_time)

    return batch_times

device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
# device = "cpu"
# model = load_model('models/CNN/2024-11-19 10:05:54/full_model.pth', device=device)
# model = load_model('models/RESNET/2024-11-19 13:13:15/full_model.pth', device=device)
model = load_model('models/ViT/2024-11-30 11:51:01/full_model.pth', device=device)

times = run_timing_eval(model, num_images=320, balance_ratio=0.50, isViT=True)
avg_time_ms = sum(times) / len(times) / 32 * 1000
print(f"Average time per image: {avg_time_ms:0.2f}ms")

# Time Trials:
# CNN
# GPU: 1.55ms per image
# CPU: 39.46ms per image

# ResNet
# GPU: 4.91ms per image
# CPU: X ms per image

# ViT
# GPU: 13.16 ms per image
# CPU: X ms per image