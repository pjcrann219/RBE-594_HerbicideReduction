import torch
import os
from eval_utilities import load_model, perform_evaluation, run_eval, compute_confusion_matrix, plot_confusion_matrix

def evaluate_models():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Define model paths
    models_to_evaluate = [
        ("RESNET Full Model", "models/RESNET/full_model.pth"),
        ("RESNET State Dict", "models/RESNET/state_dict.pth"),
        ("CNN Full Model", "models/CNN/full_model.pth"),
        ("CNN State Dict", "models/CNN/state_dict.pth")
    ]

    # Check if model files exist
    print("\nChecking model paths...")
    for model_name, path in models_to_evaluate:
        if os.path.exists(path):
            print(f"Found {model_name} at: {path}")
            print(f"File size: {os.path.getsize(path) / (1024*1024):.2f} MB")
        else:
            print(f"ERROR: {model_name} not found at: {path}")

    try:
        # Evaluate each model
        for model_name, model_path in models_to_evaluate:
            if os.path.exists(model_path):
                print(f"\n=== Evaluating {model_name} ===")
                try:
                    model = load_model(model_path, device=device)
                    
                    # Full evaluation with ROC curve
                    perform_evaluation(model, subtitle=model_name)
                    
                    # Additional confusion matrix at specific threshold
                    results = run_eval(model, num_images=3200)
                    cm = compute_confusion_matrix(results, threshold=0.5)
                    plot_confusion_matrix(cm, subtitle=f'{model_name} (threshold=0.5)')
                except Exception as e:
                    print(f"Error evaluating {model_name}: {str(e)}")
                    continue

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nPlease verify:")
        print("1. Model files exist in the correct location")
        print("2. Model files are valid PyTorch models")
        print("3. You're running the script from the correct directory")

if __name__ == "__main__":
    evaluate_models()
