import torch
from models import CNN_512_4
from Utilities import *
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from matplotlib import patches
import seaborn as sns

# Loads a PyTorch model from a file path and moves it to the specified device
def load_model(model_path, device=False):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    return model

# Evaluates model performance on test data and returns results as a DataFrame
def run_eval(model, num_images=32):
    all_data = []
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Agriculture-Vision-2021/val")
    test_loader = get_dataloader(
        test_dir,
        batch_size=32,
        num_workers=0,
        shuffle=False,
        num_images=num_images
    )
    device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            outputs = outputs.cpu().numpy() if torch.is_tensor(outputs) else outputs
            labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
            all_data.extend([
                {'label': label, 'output': output}
                for label, output in zip(labels, outputs)
            ])
    results = pd.DataFrame(all_data)
    return results

# Calculates confusion matrix based on predictions using given threshold
def compute_confusion_matrix(data, threshold):
    predictions = np.array([1 if x[0] > threshold else 0 for x in data['output']])
    cm = confusion_matrix(data['label'], predictions)
    return cm

# Calculates True Positive Rate and False Positive Rate from confusion matrix
def compute_rates(cm):
    TPR = cm[0, 1] / (cm[0, 1] + cm[1, 0])
    FPR = cm[0, 1] / (cm[0, 1] + cm[0, 0])
    return TPR, FPR

# Computes ROC curve data points across different thresholds
def compute_ROC(data, threholds=False):
    TPRs = []
    FPRs = []
    thresholds = [0, .001, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    for threshold in thresholds:
        cm = compute_confusion_matrix(data, threshold=threshold)
        TPR, FPR = compute_rates(cm)
        TPRs.append(TPR)
        FPRs.append(FPR)
    return thresholds, TPRs, FPRs

# Visualizes ROC curve with threshold labels and AUC score
def plot_ROC(thresholds, TPRs, FPRs):
    AUC = compute_AUC(TPRs, FPRs)
    FPR_Target = patches.Rectangle((0, 0), 0.2, 1.0, linewidth=1, edgecolor='green', facecolor='green', alpha=0.5, label='Target FPR')
    TPR_Target = patches.Rectangle((0, 0.95), 1.0, 0.05, linewidth=1, edgecolor='green', facecolor='green', alpha=0.5, label='Target TPR')
    fig, ax = plt.subplots(figsize=(6,5))
    ax.add_patch(FPR_Target)
    ax.add_patch(TPR_Target)
    ax.plot(FPRs, TPRs, '.-')
    for i in range(len(FPRs)):
        ax.text(FPRs[i], TPRs[i], f'{thresholds[i]:.3f}', fontsize=6, color='red', ha='right', va='bottom')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.grid()
    ax.set_title(f'Reciever Operating Characteristic Curve\nAUC: {AUC:.3f}')
    ax.legend()
    plt.show()

# Visualizes confusion matrix as a heatmap
def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, 
                xticklabels=['False', 'True'], yticklabels=['False', 'True'])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.show()

# Calculates Area Under Curve (AUC) for ROC curve
def compute_AUC(TPRs, FPRs):
    sorted_indices = np.argsort(FPRs)
    FPRs_sorted = np.array(FPRs)[sorted_indices]
    TPRs_sorted = np.array(TPRs)[sorted_indices]
    auc = np.trapz(TPRs_sorted, FPRs_sorted)
    
    return auc

# Runs complete model evaluation pipeline and displays ROC curve
def perform_evaluation(model):
    results = run_eval(model, num_images=320)
    thresholds, TPRs, FPRs = compute_ROC(results)
    plot_ROC(thresholds, TPRs, FPRs)

device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
model = load_model('models/full_model.pth', device=device)
perform_evaluation(model)