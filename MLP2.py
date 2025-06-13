import numpy as np
import scipy as sp
import scipy.io.wavfile as wav
import pandas as pd
import matplotlib.pyplot as plt
from fractions import Fraction
from sklearn.decomposition import PCA
from sklearn import datasets as SKdata
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold as KfCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import pickle
from itertools import product
from utils import read_csv
from time import time
import os
import warnings
warnings.filterwarnings('ignore')

#%%
# Parameters:
Tw = 0.025
Tstep = 0.010
Fs = 16e3
Nw = int(np.ceil(Tw*Fs))                   # frame size
Nstep = int(np.ceil(Tstep*Fs))             # frame step size
Nfft = int(pow(2, np.ceil(np.log2(Nw))))   # no. of FFT points

#%%

# Improved MLP Model with Batch Normalization
class ImprovedMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super(ImprovedMLP, self).__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for i, size in enumerate(hidden_sizes):
            # Linear layer
            self.layers.append(nn.Linear(prev_size, size))
            
            # Batch normalization (except for last hidden layer)
            if i < len(hidden_sizes) - 1:
                self.layers.append(nn.BatchNorm1d(size))
            
            # Activation function
            self.layers.append(nn.LeakyReLU(negative_slope=0.1))
            
            # Dropout
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = size
            
        self.output_layer = nn.Linear(prev_size, output_size)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x  # No softmax here, CrossEntropyLoss includes it

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig('plots/last_cf_matrix.png')
    plt.close()

def calculate_class_weights(y_train):
    """Calculate class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return torch.FloatTensor(weights)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, num_epochs=500, patience=30):
    """Improved training function with better early stopping and monitoring"""
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        
        # Print progress
        if epoch % 10 == 0 or epoch < 10:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses, val_accuracies, best_val_acc

def evaluate_model(model, data_loader, device):
    """Improved model evaluation"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    acc = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    return acc, f1, all_targets, all_predictions

if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Convert data to tensors and send to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on: ' + str(device))

    # Improved hyperparameters
    hidden_layers = [
        [64, 32],           # Simpler architectures work better for small datasets
        [32, 16],
        [128, 32],
        [64, 32, 16],
        [32, 16, 8]
    ]
    dropouts = [0.1, 0.2, 0.3, 0.4]  # Lower dropout values
    output_size = 2
    batch_size = 64  # Increased from 32
    num_epochs = 200  # Reduced from 10000
    learning_rate = 0.005  # Increased from 0.001
    input_size = 3
    patience = 30  # Early stopping patience

    # Saving best model
    best_model = None
    best_val_score = -float('inf')
    
    Nsim = len(hidden_layers) * len(dropouts)
    Kclass = 2
    windows = [25]
    window_types = ['hamming']

    for window, window_type in zip(windows, window_types):
        idx_sim = 0
        splits = [0, 1, 2, 3, 4]  # From 0 to 4 for each split
        file_paths = [f"split_data/fold_{i}_with_split.csv" for i in splits]
        METRIX_ = np.zeros((Nsim, 4))

        for hidden_sizes in hidden_layers:
            for dropout in dropouts:
                METRIX = []
                print(f"\n=== Testing architecture: {hidden_sizes}, dropout: {dropout} ===")
                
                for split_index, file_path in enumerate(file_paths):
                    print(f"\nProcessing split {split_index + 1}/5")
                    
                    # Read data
                    X_train_split, Y_train, X_val_split, Y_val = read_csv(file_path)

                    print(f"X_train_split shape: {X_train_split.shape}")
                    print(f"X_val_split shape: {X_val_split.shape}")

                    # Shuffle data
                    X_train_split, Y_train = shuffle(X_train_split, Y_train, random_state=42)

                    # Feature scaling - IMPORTANT for neural networks
                    scaler = StandardScaler()
                    X_train_split = scaler.fit_transform(X_train_split)
                    X_val_split = scaler.transform(X_val_split)

                    # Calculate class weights for imbalanced data
                    class_weights = calculate_class_weights(Y_train).to(device)

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
                    MODEL = ImprovedMLP(input_size, hidden_sizes, output_size, dropout).to(device)
                    
                    # Weighted loss for class imbalance
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                    
                    # Improved optimizer with weight decay
                    optimizer = torch.optim.AdamW(MODEL.parameters(), lr=learning_rate, 
                                                weight_decay=0.01, betas=(0.9, 0.999))
                    
                    # Learning rate scheduler
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                patience=10)

                    # Training
                    train_losses, val_losses, val_accuracies, best_val_acc = train_model(
                        MODEL, train_loader, val_loader, criterion, optimizer, scheduler,
                        device, num_epochs=num_epochs, patience=patience
                    )

                    # Plot Loss Graph
                    plt.figure(figsize=(12, 4))
                    
                    plt.subplot(1, 2, 1)
                    plt.plot(train_losses, label='Train Loss', alpha=0.8)
                    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Training and Validation Loss')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    plt.subplot(1, 2, 2)
                    plt.plot(val_accuracies, label='Validation Accuracy', color='green', alpha=0.8)
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.title('Validation Accuracy')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    os.makedirs('plots', exist_ok=True)
                    output_plot_path = f"plots/{'-'.join(map(str, hidden_sizes))}_d{dropout}_split_{split_index}.png"
                    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    # Evaluation
                    acc_train, f1_train, train_targets, train_predictions = evaluate_model(MODEL, train_loader, device)
                    acc_val, f1_val, val_targets, val_predictions = evaluate_model(MODEL, val_loader, device)

                    # Plot confusion matrices
                    plot_confusion_matrix(train_targets, train_predictions, "Confusion Matrix - Train")
                    plot_confusion_matrix(val_targets, val_predictions, "Confusion Matrix - Validation")

                    # Print results
                    params_string = f"{'-'.join(map(str, hidden_sizes))}_d{dropout}"
                    print(f'\n{params_string}')
                    print(f'acc (train) = {acc_train:.4f}. f1 (train) = {f1_train:.4f}')
                    print(f'acc (val) = {acc_val:.4f}. f1 (val) = {f1_val:.4f}')

                    METRIX.extend([acc_train, f1_train, acc_val, f1_val])

                # Calculate cross-validation averages
                acc_train_avg = np.mean([METRIX[i] for i in range(0, len(METRIX), 4)])
                f1_train_avg = np.mean([METRIX[i] for i in range(1, len(METRIX), 4)])
                acc_val_avg = np.mean([METRIX[i] for i in range(2, len(METRIX), 4)])
                f1_val_avg = np.mean([METRIX[i] for i in range(3, len(METRIX), 4)])

                print(f'\n=== CROSS-VALIDATION RESULTS ===')
                print(f'Acc avg (train) = {acc_train_avg:.4f}. F1 avg (train) = {f1_train_avg:.4f}')
                print(f'Acc avg (val) = {acc_val_avg:.4f}. F1 avg (val) = {f1_val_avg:.4f}')
                
                METRIX_[idx_sim, :] = [acc_train_avg, f1_train_avg, acc_val_avg, f1_val_avg]

                # Update best model if current is better
                if f1_val_avg > best_val_score:
                    best_val_score = f1_val_avg
                    best_model = MODEL
                    print(f"🎉 New best model found!")
                    print(f"Architecture: {hidden_sizes}, Dropout: {dropout}")
                    print(f"Val F1 Score: {best_val_score:.4f}")

                    # Save best model
                    file_path = f"models/best_mlp_{window}_{'-'.join(map(str, hidden_sizes))}_d{dropout}_{acc_val_avg:.3f}_{f1_val_avg:.3f}.pkl"
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    with open(file_path, 'wb') as f:
                        pickle.dump({
                            'model_state_dict': best_model.state_dict(),
                            'architecture': hidden_sizes,
                            'dropout': dropout,
                            'scaler': scaler,  # Save scaler for future predictions
                            'metrics': {
                                'acc_train': acc_train_avg,
                                'f1_train': f1_train_avg,
                                'acc_val': acc_val_avg,
                                'f1_val': f1_val_avg
                            }
                        }, f)
                
                idx_sim += 1

        # Save results to CSV
        sim_list_idx = range(0, Nsim)
        sim_list_hiddens = []
        sim_list_dropouts = []
        for hid in hidden_layers:
            for dropout in dropouts:
                sim_list_hiddens.append('-'.join(map(str, hid)))
                sim_list_dropouts.append(dropout)

        df_dict = {
            'SIM': sim_list_idx,
            'Architecture': sim_list_hiddens,
            'Dropout': sim_list_dropouts,
            'Acc_train': np.round(METRIX_[:, 0], 4),
            'F1_train': np.round(METRIX_[:, 1], 4),
            'Acc_val': np.round(METRIX_[:, 2], 4),
            'F1_val': np.round(METRIX_[:, 3], 4)
        }
        
        df = pd.DataFrame(df_dict)
        df = df.sort_values('F1_val', ascending=False)  # Sort by validation F1 score
        
        results_path = f'FCNN_Xval_{window}_hamming_improved.csv'
        df.to_csv(results_path, index=False)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Results saved to: {results_path}")
        print(f"Best model F1 score: {best_val_score:.4f}")
        print("\nTop 5 configurations:")
        print(df.head().to_string(index=False))
