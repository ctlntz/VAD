import numpy as np
import scipy as sp
import scipy.io.wavfile as wav
import pandas as pd
import matplotlib.pyplot as plt
from fractions import Fraction
# from python_speech_features import mfcc
from sklearn.decomposition import PCA
from sklearn import datasets as SKdata
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold as KfCV
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from itertools import product
from utils import read_csv
from time import time
import os

#%%
# Parameters:
Tw = 0.025
Tstep = 0.010
Fs = 16e3
Nw = int(np.ceil(Tw*Fs))                   # frame size
Nstep = int(np.ceil(Tstep*Fs))             # frame step size
Nfft = int(pow(2, np.ceil(np.log2(Nw))))   # no. of FFT points

#%%
# Define MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.LeakyReLU(negative_slope=0.1))
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = size
            
        self.output_layer = nn.Linear(prev_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return self.softmax(x)

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig('plots/last_cf_matrix.png')

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics including weighted and unweighted accuracy"""
    weighted_acc = accuracy_score(y_true, y_pred)  # Standard accuracy (weighted by support)
    unweighted_acc = balanced_accuracy_score(y_true, y_pred)  # Unweighted accuracy (macro average)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    unweighted_f1 = f1_score(y_true, y_pred, average='macro')
    
    return {
        'weighted_acc': weighted_acc,
        'unweighted_acc': unweighted_acc,
        'weighted_f1': weighted_f1,
        'unweighted_f1': unweighted_f1
    }

if __name__ == '__main__':
    # Convert data to tensors and send to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on: ' + str(device))

    # Initialize best model tracking
    best_models_info = []  # List to store all best model configurations
    overall_best_model = None
    overall_best_score = -float('inf')
    overall_best_config = None

    hidden_layers = [
                    #  [128, 64, 32, 16], 
                    #  [128, 32, 16, 16], 
                    #  [64, 32, 16], 
                    #  [64, 32, 8], 
                    #  [32, 16, 16], 
                    #  [32, 16, 8], 
                    #  [64, 32], 
                    #  [32, 16], 
                     [32, 8]
                    ]
    dropouts = [0.2
                # , 0.3, 0.4, 0.5
                ]
    output_size = 2
    batch_size = 16
    num_epochs = 100
    learning_rate = 0.001
    input_size = 15

    Nsim = len(hidden_layers)*len(dropouts)
    Kclass = 2
    windows = [25]
    window_types = ['hamming']

    # Create directories for saving models and results
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    for window, window_type in zip(windows, window_types):
        idx_sim = 0
        splits = [0, 1, 2, 3, 4]  # From 0 to 4 for each split
        file_paths = [f"split_data/fold_{i}_with_splits.csv" for i in splits]
        # Extended metrics matrix to include unweighted metrics
        METRIX_ = np.zeros((Nsim, 8))  # 8 metrics now: 4 weighted + 4 unweighted

        for hidden_sizes in hidden_layers:
            for dropout in dropouts:
                METRIX = []
                config_best_model = None
                config_best_score = -float('inf')
                
                for split_index, file_path in enumerate(file_paths):
                    # Read data
                    X_train_split, Y_train, X_val_split, Y_val = read_csv(file_path)

                    print(f"X_train_split shape: {X_train_split.shape}")
                    print(f"X_val_split shape: {X_val_split.shape}")

                    # Shuffle data
                    X_train_split, Y_train = shuffle(X_train_split, Y_train, random_state=42)

                    # Define the tensors
                    X_train_tensor = torch.tensor(X_train_split, dtype=torch.float32).to(device)
                    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long).to(device)
                    X_val_tensor = torch.tensor(X_val_split, dtype=torch.float32).to(device)
                    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long).to(device)

                    # Create DataLoaders
                    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
                    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                    # Model, Loss, Optimizer
                    MODEL = MLP(input_size, hidden_sizes, output_size, dropout).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(MODEL.parameters(), lr=0.0001, weight_decay=1e-2)

                    # Training Loop with Loss Tracking
                    train_losses = []
                    val_losses = []

                    # Early stopping parameters
                    early_stop_patience = 10  # Number of epochs to wait for improvement
                    best_val_loss = float('inf')
                    epochs_no_improve = 0

                    for epoch in range(num_epochs):
                        iter_time = []
                        MODEL.train()
                        epoch_loss = 0.0
                        for batch_idx, (inputs, targets) in enumerate(train_loader):
                            optimizer.zero_grad()
                            start = time()
                            outputs = MODEL(inputs)
                            if outputs.shape[1] != Kclass:
                                raise ValueError(f"Output shape {outputs.shape[1]} does not match number of classes {Kclass}.")

                            if targets.max().item() >= Kclass:
                                raise ValueError(f"Target value {targets.max().item()} is out of bounds for the number of classes {Kclass}.")
                            
                            stop = time()
                            iter_time.append(stop-start)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()

                        train_losses.append(epoch_loss / len(train_loader))
                        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}')

                        # Evaluate on validation set every few epochs
                        if (epoch) % 5 == 0:
                            MODEL.eval()
                            val_loss = 0.0
                            val_batches = 0
                            
                            try:
                                with torch.inference_mode():
                                    for inputs, targets in val_loader:
                                        outputs = MODEL(inputs)
                                        loss = criterion(outputs, targets)
                                        val_loss += loss.item()
                                        val_batches += 1
                                
                                if val_batches > 0:
                                    val_losses.append(val_loss / val_batches)
                                    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
                                else:
                                    print(f"Warning: No validation batches processed in epoch {epoch+1}")
                                    if val_losses:
                                        val_losses.append(val_losses[-1])
                                    else:
                                        val_losses.append(train_losses[-1])
                                    
                            except Exception as e:
                                print(f"Error during validation: {e}")
                                if val_losses:
                                    val_losses.append(val_losses[-1])
                                else:
                                    val_losses.append(train_losses[-1])

                            # Check early stopping condition
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                epochs_no_improve = 0
                            else:
                                epochs_no_improve += 1

                            if epochs_no_improve >= early_stop_patience:
                                print(f"Early stopping triggered at epoch {epoch + 1}")
                                break

                    # Plot Loss Graph
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
                    plt.plot(range(10, 10 * len(val_losses) + 1, 10), val_losses, label='Validation Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Training and Validation Loss')
                    plt.legend()
                    plt.grid(True)
                    output_plot_path = f"plots/{' '.join(map(str, hidden_sizes))}_split_{split_index}.png"
                    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    # Evaluation with comprehensive metrics
                    MODEL.eval()
                    with torch.inference_mode():
                        # Process training data
                        train_outputs = []
                        train_targets = []
                        for inputs, targets in train_loader:
                            outputs = MODEL(inputs).cpu().numpy()
                            train_outputs.append(outputs)
                            train_targets.append(targets.cpu().numpy())
                        
                        if train_outputs:
                            train_outputs = np.concatenate(train_outputs)
                            train_targets = np.concatenate(train_targets)
                            train_predictions = np.argmax(train_outputs, axis=1)
                            
                            # Calculate comprehensive training metrics
                            train_metrics = calculate_metrics(train_targets, train_predictions)
                            plot_confusion_matrix(train_targets, train_predictions, "Confusion Matrix - Train")
                        else:
                            print("Warning: No training data available for evaluation")
                            train_metrics = {k: 0.0 for k in ['weighted_acc', 'unweighted_acc', 'weighted_f1', 'unweighted_f1']}
                        
                        # Process validation data
                        val_outputs = []
                        val_targets = []
                        val_loader_empty = True
                        
                        for inputs, targets in val_loader:
                            val_loader_empty = False
                            if targets.dim() > 1:
                                targets = targets.argmax(dim=1)
                            outputs = MODEL(inputs).cpu().numpy()
                            val_outputs.append(outputs)
                            val_targets.append(targets.cpu().numpy())
                        
                        if not val_loader_empty and val_outputs:
                            val_outputs = np.concatenate(val_outputs)
                            val_targets = np.concatenate(val_targets)
                            val_predictions = np.argmax(val_outputs, axis=1)
                            
                            # Calculate comprehensive validation metrics
                            val_metrics = calculate_metrics(val_targets, val_predictions)
                            plot_confusion_matrix(val_targets, val_predictions, "Confusion Matrix - Validation")
                        else:
                            print("Warning: No validation data available for evaluation")
                            val_metrics = {k: 0.0 for k in ['weighted_acc', 'unweighted_acc', 'weighted_f1', 'unweighted_f1']}

                    # Print results with new metrics
                    params_string = '_'.join(list(map(str, [hidden_sizes] + [dropout])))
                    print(f'\n{params_string}')
                    print(f'Train - Weighted Acc: {train_metrics["weighted_acc"]:.4f}, Unweighted Acc: {train_metrics["unweighted_acc"]:.4f}')
                    print(f'Train - Weighted F1: {train_metrics["weighted_f1"]:.4f}, Unweighted F1: {train_metrics["unweighted_f1"]:.4f}')
                    print(f'Val - Weighted Acc: {val_metrics["weighted_acc"]:.4f}, Unweighted Acc: {val_metrics["unweighted_acc"]:.4f}')
                    print(f'Val - Weighted F1: {val_metrics["weighted_f1"]:.4f}, Unweighted F1: {val_metrics["unweighted_f1"]:.4f}')

                    # Store all metrics
                    METRIX.extend([
                        train_metrics["weighted_acc"], train_metrics["unweighted_acc"],
                        train_metrics["weighted_f1"], train_metrics["unweighted_f1"],
                        val_metrics["weighted_acc"], val_metrics["unweighted_acc"],
                        val_metrics["weighted_f1"], val_metrics["unweighted_f1"]
                    ])
                    
                    # Check if this is the best model for this configuration
                    current_score = val_metrics["unweighted_f1"]  # Using unweighted F1 as primary metric
                    if current_score > config_best_score:
                        config_best_score = current_score
                        config_best_model = MODEL.state_dict().copy()

                # Calculate averages across splits
                L = len(METRIX)
                averages = []
                for metric_idx in range(8):  # 8 metrics now
                    metric_sum = sum(METRIX[i] for i in range(metric_idx, L, 8))
                    averages.append(np.round(metric_sum / 5, decimals=4))
                
                train_weighted_acc_avg, train_unweighted_acc_avg = averages[0], averages[1]
                train_weighted_f1_avg, train_unweighted_f1_avg = averages[2], averages[3]
                val_weighted_acc_avg, val_unweighted_acc_avg = averages[4], averages[5]
                val_weighted_f1_avg, val_unweighted_f1_avg = averages[6], averages[7]

                print(f'\nConfiguration Results:')
                print(f'Train - Weighted Acc: {train_weighted_acc_avg:.4f}, Unweighted Acc: {train_unweighted_acc_avg:.4f}')
                print(f'Train - Weighted F1: {train_weighted_f1_avg:.4f}, Unweighted F1: {train_unweighted_f1_avg:.4f}')
                print(f'Val - Weighted Acc: {val_weighted_acc_avg:.4f}, Unweighted Acc: {val_unweighted_acc_avg:.4f}')
                print(f'Val - Weighted F1: {val_weighted_f1_avg:.4f}, Unweighted F1: {val_unweighted_f1_avg:.4f}\n')
                
                METRIX_[idx_sim, :] = averages

                # Save best model for this configuration and track it
                model_info = {
                    'hidden_layers': hidden_sizes,
                    'dropout': dropout,
                    'window': window,
                    'train_weighted_acc': train_weighted_acc_avg,
                    'train_unweighted_acc': train_unweighted_acc_avg,
                    'train_weighted_f1': train_weighted_f1_avg,
                    'train_unweighted_f1': train_unweighted_f1_avg,
                    'val_weighted_acc': val_weighted_acc_avg,
                    'val_unweighted_acc': val_unweighted_acc_avg,
                    'val_weighted_f1': val_weighted_f1_avg,
                    'val_unweighted_f1': val_unweighted_f1_avg,
                    'primary_score': val_unweighted_f1_avg  # Using unweighted F1 as primary metric
                }
                
                # Save the best model for this configuration
                model_filename = f"models/best_mlp_{window}_{'_'.join(map(str, hidden_sizes))}_{dropout}_{val_unweighted_f1_avg:.4f}.pkl"
                with open(model_filename, 'wb') as f:
                    pickle.dump(config_best_model, f)
                
                model_info['model_path'] = model_filename
                best_models_info.append(model_info)
                
                # Check if this is the overall best model
                if val_unweighted_f1_avg > overall_best_score:
                    overall_best_score = val_unweighted_f1_avg
                    overall_best_model = config_best_model
                    overall_best_config = model_info.copy()
                    print(f"New overall best model: Hidden={hidden_sizes}, Dropout={dropout}, Score={overall_best_score:.4f}")

                idx_sim += 1

        # Save comprehensive results to CSV
        sim_list_idx = range(0, Nsim)
        sim_list_hiddens = []
        sim_list_dropouts = []
        for hid in hidden_layers:
            for dropout in dropouts:
                sim_list_hiddens.append('_'.join(map(str, hid)))
                sim_list_dropouts.append(dropout)

        df_dict = {
            'SIM': sim_list_idx,
            'Hidden_Layers': sim_list_hiddens,
            'Dropout': sim_list_dropouts,
            'Train_Weighted_Acc': METRIX_[:, 0],
            'Train_Unweighted_Acc': METRIX_[:, 1],
            'Train_Weighted_F1': METRIX_[:, 2],
            'Train_Unweighted_F1': METRIX_[:, 3],
            'Val_Weighted_Acc': METRIX_[:, 4],
            'Val_Unweighted_Acc': METRIX_[:, 5],
            'Val_Weighted_F1': METRIX_[:, 6],
            'Val_Unweighted_F1': METRIX_[:, 7]
        }
        
        df = pd.DataFrame(df_dict)
        results_path = f'results/FCNN_Xval_{window}_hamming_comprehensive.csv'
        df.to_csv(results_path, index=False)
        print(f"Comprehensive results saved to: {results_path}")

    # Save best models configuration summary
    best_models_df = pd.DataFrame(best_models_info)
    best_models_df = best_models_df.sort_values('primary_score', ascending=False)
    best_models_path = 'results/best_models_summary.csv'
    best_models_df.to_csv(best_models_path, index=False)
    print(f"Best models summary saved to: {best_models_path}")

    # Save the overall best model separately
    if overall_best_model is not None:
        overall_best_path = f"models/overall_best_model_{overall_best_score:.4f}.pkl"
        with open(overall_best_path, 'wb') as f:
            pickle.dump(overall_best_model, f)
        
        # Save overall best configuration
        with open('results/overall_best_config.txt', 'w') as f:
            f.write("Overall Best Model Configuration:\n")
            f.write("="*50 + "\n")
            f.write(f"Hidden Layers: {overall_best_config['hidden_layers']}\n")
            f.write(f"Dropout: {overall_best_config['dropout']}\n")
            f.write(f"Window: {overall_best_config['window']}\n")
            f.write(f"Primary Score (Val Unweighted F1): {overall_best_config['primary_score']:.4f}\n")
            f.write(f"Model Path: {overall_best_path}\n")
            f.write("\nDetailed Metrics:\n")
            f.write(f"Train Weighted Acc: {overall_best_config['train_weighted_acc']:.4f}\n")
            f.write(f"Train Unweighted Acc: {overall_best_config['train_unweighted_acc']:.4f}\n")
            f.write(f"Train Weighted F1: {overall_best_config['train_weighted_f1']:.4f}\n")
            f.write(f"Train Unweighted F1: {overall_best_config['train_unweighted_f1']:.4f}\n")
            f.write(f"Val Weighted Acc: {overall_best_config['val_weighted_acc']:.4f}\n")
            f.write(f"Val Unweighted Acc: {overall_best_config['val_unweighted_acc']:.4f}\n")
            f.write(f"Val Weighted F1: {overall_best_config['val_weighted_f1']:.4f}\n")
            f.write(f"Val Unweighted F1: {overall_best_config['val_unweighted_f1']:.4f}\n")
        
        print(f"Overall best model saved to: {overall_best_path}")
        print(f"Overall best configuration saved to: results/overall_best_config.txt")
        print(f"Overall best score: {overall_best_score:.4f}")

    print("\nTraining completed! All models and results have been saved.")