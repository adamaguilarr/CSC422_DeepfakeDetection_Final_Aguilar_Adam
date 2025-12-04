# CSC422 Final Project – Deepfake Frame Classification

Author: Adam Aguilar  
Course: CSC422 Machine and Deep Learning  

This project builds a deep learning model that classifies individual video frames as **REAL** or **FAKE** using transfer learning with ResNet18. The work is based on the DeepFake Detection Challenge (DFDC) sample dataset and is organized into three Jupyter notebooks:

1. `01_data_exploration.ipynb` – Explore the raw DFDC videos and metadata.
2. `02_model_development.ipynb` – Extract frames, build the dataset, and train a ResNet18 classifier.
3. `03_results_analysis.ipynb` – Load the trained model, evaluate on a validation split, and visualize results.

The final model reaches about **92 percent validation accuracy** on single frames and is especially strong at detecting fake images.

---

## 1. Project structure

```text
CSC422_DeepfakeDetection_Final_Aguilar_Adam/
├─ data/
│  ├─ raw/
│  │  ├─ train_sample_videos/        # Original DFDC videos (mp4)
│  │  ├─ test_videos/                # Optional test videos
│  │  └─ metadata.json               # Labels and split info from DFDC
│  └─ processed/
│     └─ frames/
│        └─ train/
│           ├─ REAL/                 # Extracted frames for real videos
│           └─ FAKE/                 # Extracted frames for fake videos
├─ models/
│  └─ deepfake_resnet18.pth          # Trained ResNet18 weights
├─ notebooks/
│  ├─ 01_data_exploration.ipynb
│  ├─ 02_model_development.ipynb
│  └─ 03_results_analysis.ipynb
├─ results/
│  ├─ figures/
│  │  ├─ confusion_matrix.png        # Validation confusion matrix
│  │  └─ metrics_bar.png             # Example bar chart of metrics
│  └─ metrics/
│     ├─ eval_metrics.json           # Stored accuracy and other metrics
│     └─ eval_predictions.csv        # Predictions for each validation frame
├─ src/                              # (Optional) helper modules, if used
├─ .gitignore
├─ environment.yml (optional)
├─ requirements.txt
└─ README.md