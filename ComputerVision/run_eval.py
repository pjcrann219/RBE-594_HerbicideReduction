from eval_utilities import *
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
model = load_model('models/CNN/2024-11-19 10:05:54/full_model.pth', device=device)
perform_evaluation(model, subtitle='CNN Model', show=False)
plt.savefig('imgs/CNN_ROC.png')
plt.close()

device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
model = load_model('models/RESNET/full_model.pth', device=device)
perform_evaluation(model, subtitle='RESNET50 Model', show=False)
plt.savefig('imgs/RESNET_ROC.png')
plt.close()

device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
model = load_model('models/ViT/2024-11-30 11:51:01/full_model.pth', device=device)
perform_evaluation(model, subtitle='ViT Model', isViT=True, show=False)
plt.savefig('imgs/ViT_ROC.png')
plt.close()

# device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
# model = load_model('models/CNN/full_model.pth', device=device)
# perform_evaluation(model, subtitle='Custom CNN Model')

# device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
# model = load_model('models/RESNET/full_model2.pth', device=device)
# results = run_eval(model, num_images=3200)
# cm = compute_confusion_matrix(results, threshold=0.1)
# plot_confusion_matrix(cm, subtitle='RESNET50 Threshold = 0.1')

# device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
# model = load_model('models/CNN/full_model.pth', device=device)
# results = run_eval(model, num_images=3200)
# cm = compute_confusion_matrix(results, threshold=0.1)
# plot_confusion_matrix(cm, subtitle='Custom CNN Threshold = 0.1')

# device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
# model = load_model('models/ViT/2024-11-30 11:51:01/full_model.pth', device=device)
# results = run_eval(model, num_images=3200, isViT=True)
# cm = compute_confusion_matrix(results, threshold=0.1)
# plot_confusion_matrix(cm, subtitle='ViT Threshold = 0.1')