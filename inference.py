import os
import numpy as np
import torch
from data_processing import *
from utils import *
from time import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils import shuffle
from mlp import MLP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm

# Num channels 
NUM_CHANNELS = 7
# Use freq features
FREQ_FEAT = False 

### mlp params dict ###
mlp_params = {
    "input_size": 56,
    "hidden_layers": [128, 64, 64, 32, 16],
    "dropout": 0.3,
    "output_size": 3,
}
########################

### rf params dict ###
rf_params = {
    'n_estimators': 5,
    'min_samples_split': 0.1,
    'max_depth': None,
    'min_samples_leaf': 0.05,
    'max_samples': 0.8,
    'max_features': 'sqrt'
}
########################

### rf params dict ###
svm_params = {
    'pca_comp': 'no_pca',
    'C': 100,
    'SVM_kernel': 'rbf',
    'tol': 1
    # others
}
########################

#### mlp utils
def load_mlp(model_path, device):
    model = MLP(mlp_params["input_size"], mlp_params['hidden_layers'], mlp_params["output_size"],mlp_params['dropout']).to(device)
    if os.path.exists(model_path):
        try:  
            model.load_state_dict(torch.load(model_path, map_location=device))
        except:
            print("Could not load the model")
    return model

def train_mlp(model_path, device):
    model = load_mlp(model_path, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    num_epochs = 200

    # Define the tensors
    X_train_tensor = torch.tensor(X_train_split, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long).to(device)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)

    for _ in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    return model
    

def read_channel_statistics(folder_path, ch_stats_path, fe_stats_path, fs=512, low=20, high=250):
    if os.path.exists(ch_stats_path) and os.path.exists(fe_stats_path):
        return np.load(ch_stats_path), np.load(fe_stats_path)
    else:
        data = [[] for _ in range(NUM_CHANNELS)]
        feature_data = []
        files = os.listdir(folder_path)
        
        for file in files:
            # Read the data
            raw_data = np.load(os.path.join(folder_path, file))

            # Remove channel 4 -> no data for some users
            keep_ch = [0, 1, 2, 3, 5, 6, 7]
            raw_data = raw_data[keep_ch, :]

            # Bandpass filtering
            filtered_data = bandpass_filter(raw_data, lowcut=low, highcut=high, fs=fs, order=4)

            # Notch filtering
            filtered_data = notch_filter(filtered_data)

            # Append the data to the channel list
            for i in range(filtered_data.shape[0]):
                data[i].extend(filtered_data[i].tolist())

            # Extract features and store
            features = calculate_features(filtered_data, FREQ_FEAT)
            feature_data.append(features)
        
        # Compute channel statistics
        ch_statistics = [(np.mean(channel), np.std(channel)) for channel in data]
        np.save(ch_stats_path, np.array(ch_statistics))
        
        # Compute feature statistics
        feature_data = np.array(feature_data)
        fe_statistics = [(np.mean(feature_data[:, i]), np.std(feature_data[:, i])) for i in range(feature_data.shape[1])]
        np.save(fe_stats_path, np.array(fe_statistics))     
        return ch_statistics, fe_statistics


if __name__ == "__main__":
    window_size = 1200
    model_type = 'SVM'
    folder_path = "db/Hand_without_test"
    test_subjects_path = "db/Hand_with_test"
    ch_stats_file = f"stats_files/ch_stats_file_{window_size}.npy"
    fe_stats_file = f"stats_files/fe_stats_file_{window_size}.npy"
    output_path = "results/"
    # model_path = f"models/{model_type}/best_{model_type}_{window_size}.pkl"
    model_path = f"models/{model_type}_{window_size}_entire_set.pth"
    train_model = False
    seq_windows = 3
    sliding_window = True
    pca = None

    # Retrain the model on the entire dataset if needed    
    if train_model:
        # Train the RF
        file_path = f"db/semg_all_train_{window_size}.csv"
        X_train_split, Y_train, _, _ = read_csv(file_path)

        # Shuffle data
        X_train_split, Y_train = shuffle(X_train_split, Y_train, random_state=24)

        # Create the model - set parameters according to your configuration
        if model_type == 'RF':
            model = RF(n_estimators=rf_params['n_estimators'], 
                       min_samples_split=rf_params['min_samples_split'], 
                       max_depth=rf_params['max_depth'],
                       min_samples_leaf=rf_params['min_samples_leaf'], 
                       max_samples=rf_params['max_samples'], 
                       max_features=rf_params['max_features'])
            model.fit(X_train_split, Y_train)
        elif model_type == 'SVM':
            pca_comp = svm_params['pca_comp']
            if pca_comp != 'no_pca':
                pca = PCA(n_components=pca_comp)
                X_train_split = pca.fit_transform(X_train_split)
            model = SVC(C=svm_params['C'], 
                        kernel=svm_params['SVM_kernel'], 
                        tol=svm_params['tol'], 
                        probability=True)
            model.fit(X_train_split, Y_train)
        else:  ## MLP
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = train_mlp(model_path, device)
    
        # Save the model
        model_filename = os.path.join("models/", f"{model_type}_{window_size}_entire_set.pth")
        if model and model_type in ['RF', 'SVM']:
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved as {model_filename}")

        elif model and model_type == 'MLP':
            torch.save(model.state_dict(), model_filename)
            print(f"mlp model saved as {model_filename}")
        
    else:
        if model_type == 'RF' or model_type == 'SVM':
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = MLP(mlp_params["input_size"], mlp_params['hidden_layers'], mlp_params["output_size"],mlp_params['dropout']).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

    # Read the channel statistics or even compute them 
    channel_statistics, feature_statistics = read_channel_statistics(folder_path, ch_stats_file, fe_stats_file)

    # Save predictions
    y_true = []
    y_pred = []

    # Read the test subjects and perform inference on each exercise
    for file_name in os.listdir(test_subjects_path):
        if file_name.endswith(".npy"):
            # Extract subject info
            parts = file_name.split("_")
            nume, prenume, clasa, _ = parts
            id_subject = f"{nume}_{prenume}"
            print(f"Processing subject: {nume}, exercise: {clasa}.")

            # Process the test file
            subject_path = os.path.join(test_subjects_path, file_name)
            test_windows = process_subject(subject_path, channel_statistics, window_size)
            test_features = [calculate_features(window) for window in test_windows]

            # Convert to numpy array for inference
            test_features = np.array(test_features)

            # Feature standardization
            if pca is None:
                feature_statistics = np.array(feature_statistics)
                means = feature_statistics[0]  # First column: Means
                stds = feature_statistics[1]
                test_features = (test_features - means) / stds
            else:
                test_features = pca.transform(test_features)

            # Predict based on model type
            if model_type in ['RF', 'SVM']:
                logits = model.predict_proba(test_features)  # RF/SVM have predict_proba
            else:  # mlp (MLP)
                model.eval()
                test_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)
                with torch.inference_mode():
                    logits = model(test_tensor)  # Forward pass in PyTorch
                    logits = F.softmax(logits, dim=1)  # Convert to probabilities
                logits = logits.cpu().numpy()  # Convert to NumPy for consistency
            
            # If seq_windows = 1, make direct classification
            if seq_windows == 1:
                preds = np.argmax(logits, axis=1)
            else:
                logits_list = []
                if sliding_window:
                    half_seq = seq_windows // 2
                    for i in range(len(logits)):  
                        start_idx = max(0, i - half_seq)  # Ensure we don't go below 0
                        end_idx = min(len(logits), i + half_seq + 1)  # Ensure we don't exceed last index
                        
                        # Aggregate logits from previous, current, and next windows
                        seq_logits = logits[start_idx:end_idx].sum(axis=0)  
                
                        # Store the aggregated logits
                        logits_list.append(seq_logits)
                else:
                    # Ensure logits are grouped correctly, pad if needed
                    num_samples = len(logits)
                    num_groups = np.ceil(num_samples / seq_windows).astype(int)
                    
                    # Pad logits if necessary
                    if num_samples % seq_windows != 0:
                        padding_needed = seq_windows - (num_samples % seq_windows)
                        logits = np.vstack([logits, np.tile(logits[-1], (padding_needed, 1))])
                    
                    
                    for i in range(0, len(logits), seq_windows):
                        seq_logits = logits[i:i+seq_windows].sum(axis=0)
                        logits_list.append(seq_logits)

                # Determine class from grouped logits
                preds = np.argmax(np.array(logits_list), axis=1)

            y_pred.extend(preds)
            y_true.extend([int(clasa)] * len(preds))            

    # Compute classification report
    class_report = classification_report(y_true, y_pred, digits=4)
    report_filename = os.path.join(output_path, f'classification_report_{model_type}_{window_size}_{seq_windows}.txt')
    with open(report_filename, "w") as f:
        f.write("Classification Report:\n")
        print("Classification Report:")
        f.write(class_report)
        print(class_report)
    
    # Compute and plot confusion matrix as percentages
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (%)")
    plt.savefig(os.path.join(output_path, f'cnf_matrix_{model_type}_{window_size}_{seq_windows}.png'))
