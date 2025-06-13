import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import librosa
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as pBASIC
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random

from scipy.signal import butter, filtfilt
from scipy.stats import skew
from time import time
from sklearn.model_selection import KFold
from scipy.signal import iirnotch, filtfilt
from scipy import signal
from sklearn.preprocessing import StandardScaler
from scipy.signal import welch

# Global variables
FS = 16000   # Frecventa de esantionare (Hz)
WS = 25      # Window size (miliseconds)
OVR = 0.6    # Window overlap
LOW = 20
HIGH = 15800
NFFT = 512
NMFCC = 13
NORM = 'z-score'
WINDOW = 'hamming'
FILT = 'none'
CROSS_VAL_SPLIT = True  ## Whether to split the dataset or not (False for cross-validation)
N_SPLITS = 5
ALL_TRAIN = True
period_values = []
amplitude_envelopes = []

# -------------------- 1. Filtrare Bandpass Butterworth --------------------
# def butter_bandpass(lowcut, highcut, fs, order=4):
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     return b, a

# def bandpass_filter(data, lowcut=20, highcut=250, fs=FS, order=4):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     return filtfilt(b, a, data, axis=1)

# -------------------- 2. Filtrare Notch --------------------
# def notch_filter(data, f0=50, Q=30):
#     # Signal implementation
#     # b, a = signal.iirnotch(f0, Q, FS)
#     # data_filtered = signal.filtfilt(b, a, data, axis=1)

#     # Manual segmentation
#     f0 = 50    # Frequency to be notched (Hz)
#     bw = 2    # Bandwidth of the notch (Hz)# Design the Butterworth stopband 
#     low = (f0 - bw / 2) / (FS / 2)  # Lower cutoff frequency (normalized)
#     high = (f0 + bw / 2) / (FS / 2) # Upper cutoff frequency (normalized) 
#     b, a = butter(N=4, Wn=[low, high], btype='bandstop', output='ba')
#     data_filtered = signal.filtfilt(b, a, data, axis=1)
#     return data_filtered


# def resample_data(data):
#     res_rat = FS/512
#     if res_rat == 1:
#         pass
#     else:
#         # Limit denominator -> in case the values of the numerator/denominator are
#         # too high                
#         F_res_rat = Fraction(str(res_rat)).limit_denominator(1000)
#         P = F_res_rat.numerator
#         Q = F_res_rat.denominator
#         # --> FIR filter coeffs.; filter order = 1000; Hamming window:
#         # Cut out freq of the fiir filter -> 1 / max(P, Q)
#         # b_res -> Fiir quoetients
#         b_res = scipy.signal.firwin(1001, 1/max(P,Q))
#         # --> Polyphase resampling:
#         data = scipy.signal.resample_poly(data, P, Q, window=b_res)
#     return data


def get_feature_dict():
    # Init features list
    all_features = {}
    for i in range(NMFCC):
        all_features[f'mfcc_{i}'] = []
    #     all_features[f'dmfcc_{i}'] = []
    #     all_features[f'ddmfcc_{i}'] = []
    all_features.update({
        "fundamental_freq": [],
        "mean_abs_value": [],
        "zero_crossing_rate": [],
        # "slope_sign_changes": [],
        # "skewness": [],
        # "rms": [],
        # "mnf": [],
        # "mdf": [],
        "ttp": []
        # "s2": [],
        # "s3": [],
        # "s4": [],
        # "jitter": [],
        # "shimmer": []
    })
    return all_features


def feature_normalization(data_list):
    # Convert the list to a numpy array
    data_list = np.array(data_list, dtype=object)

    # Separate attributes and features
    attributes = data_list[:, :1]  # First 4 columns (non-numeric attributes)
    features = data_list[:, 1:].astype(float)  # Numeric features (convert to float)

    # Get the unique subject IDs
    subject_ids = np.unique(attributes[:, 0])

    # List to store the normalized data
    normalized_data = []

    # Loop through each subject and normalize its features
    for subject_id in subject_ids:
        # Get the indices for the current subject
        subject_indices = attributes[:, 0] == subject_id
        
        # Extract the features for the current subject
        subject_features = features[subject_indices]
        subject_attributes = attributes[subject_indices]
        
        # Initialize the scaler
        scaler = StandardScaler()
        
        # Fit the scaler only on the current subject's features and transform
        subject_features_normalized = scaler.fit_transform(subject_features)
        
        # Combine attributes and normalized features
        normalized_subject_data = np.hstack([subject_attributes, subject_features_normalized])
        
        # Append the normalized subject data to the list
        normalized_data.append(normalized_subject_data)

    # Combine all normalized subject data into a single array
    data_list_normalized = np.vstack(normalized_data)

    return data_list_normalized


# -------------------- 4. Ferestruire --------------------
def create_windows(data):
    step = int(WS * (1 - OVR) * FS / 1000)
    window_length = int(WS * FS / 1000)
    windows = []
    for start in range(0, data.shape[0] - window_length + 1, step):
        window = data[start:start + window_length]
        if WINDOW == 'hamming':
            hamming_window = np.hamming(window.shape[0])
            windows.append(window * hamming_window)
        elif WINDOW == 'rect':
            windows.append(window)
    return np.array(windows)

def median_filter_1d(signal, kernel_size: int = 7):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    padding = kernel_size // 2
    padded_signal = np.pad(signal, (padding, padding), mode='reflect')
    result = np.zeros_like(signal, dtype=float)
    
    for i in range(len(signal)):
        window = padded_signal[i:i + kernel_size]
        result[i] = np.median(window)
        
    return result.tolist()


# -------------------- 5. Calculul trasaturilor --------------------
def calculate_features(windows):
    # period_values, amplitude_envelopes = [], []
    
    # Store features for each window
    features_per_window = {
        # 'mean_abs_value': [], 
        # 'zero_crossing_rate': [],
        # 'slope_sign_changes': [],
        # 'skewness': [],
        # 'rms': [],
        'fundamental_freq': [],
        # 'mnf': [],
        # 'mdf': [],
        'ttp': []
        # 's2': [],
        # 's3': [],
        # 's4': []
    }
    
    # Initialize MFCC lists
    for i in range(NMFCC):
        features_per_window[f'mfcc{i}'] = []
    
    # Process each window
    for window in windows:
        # Temporal features
        # features_per_window['mean_abs_value'].append(np.mean(np.abs(window)))
        # features_per_window['zero_crossing_rate'].append(np.sum(np.abs(np.diff(window)) > 0.005))
        # features_per_window['slope_sign_changes'].append(sum(map(lambda x: (x >= 0).astype(float), 
        #                                                  (-np.diff(window, prepend=1)[1:-1]*np.diff(window)[1:]))))
        # features_per_window['skewness'].append(skew(window))
        # features_per_window['rms'].append(np.sqrt(np.mean(window**2)))

        # Extract pitch (F0) using FFT
        fft_result = np.fft.rfft(window, NFFT)
        magnitude = np.abs(fft_result)
        max_idx = np.argmax(magnitude[1:]) + 1
        fundamental_freq = max_idx / NFFT * FS
        features_per_window['fundamental_freq'].append(fundamental_freq)

        # Save fundamental periods for jitter
        # fundamental_period = 1 / fundamental_freq
        # period_values.append(fundamental_period)
        
        # MFCCS
        mfccs = librosa.feature.mfcc(y=window, sr=FS, n_mfcc=NMFCC, n_fft=int(WS/1000*FS))
        for i in range(NMFCC):
            features_per_window[f'mfcc{i}'].append(mfccs[i][0])  # Take first value if mfccs returns 2D array
        
        # Save max amplitude for shimmer
        # amplitude_envelopes.append(np.max(np.abs(window)))

        # Spectral domain features
        f, Pxx = welch(window, fs=FS, nperseg=len(window))
        
        # Handle potential division by zero
        # if np.sum(Pxx) > 0.005:
        #     features_per_window['mnf'].append(np.sum(f * Pxx) / np.sum(Pxx))
        # else:
        #     features_per_window['mnf'].append(0)
        
        # Median Frequency (MDF)
        # cumulative_power = np.cumsum(Pxx)
        # if np.sum(Pxx) > 0:
        #     mdf_idx = np.where(cumulative_power >= np.sum(Pxx) / 2)[0]
        #     if len(mdf_idx) > 0:
        #         features_per_window['mdf'].append(f[mdf_idx[0]])
        #     else:
        #         features_per_window['mdf'].append(0)
        # else:
        #     features_per_window['mdf'].append(0)
        
        # Total Spectral Power (TTP)
        features_per_window['ttp'].append(np.sum(Pxx))
        
        # Spectral Moments
        # if np.sum(Pxx) > 0.005:
            # mnf = np.sum(f * Pxx) / np.sum(Pxx)
    #         s2 = np.sqrt(np.sum((f - mnf) ** 2 * Pxx) / np.sum(Pxx))
    #         features_per_window['s2'].append(s2)
            
    #         if s2 > 0:
    #             features_per_window['s3'].append(np.sum((f - mnf) ** 3 * Pxx) / (s2 ** 3 * np.sum(Pxx)))
    #             features_per_window['s4'].append(np.sum((f - mnf) ** 4 * Pxx) / (s2 ** 4 * np.sum(Pxx)))
    #         else:
    #             features_per_window['s3'].append(0)
    #             features_per_window['s4'].append(0)
    #     else:
    #         features_per_window['s2'].append(0)
    #         features_per_window['s3'].append(0)
    #         features_per_window['s4'].append(0)

    # # Calculate jitter and shimmer statistics
    # jitter = np.abs(np.diff(period_values))
    # # Handle potential division by zero in shimmer calculation
    # safe_denominator = np.maximum(amplitude_envelopes[:-1], 0.005)
    # shimmer = np.abs(20 * np.log10(np.divide(amplitude_envelopes[1:], safe_denominator)))
    
    # Calculate deltas for MFCCs
    # for i in range(NMFCC):
    #     # Make sure we have a proper 2D array for delta calculation
    #     mfcc_array = np.array(features_per_window[f'mfcc{i}']).reshape(-1, 1)
    #     if mfcc_array.size > 0:
    #         delta_mfccs = librosa.feature.delta(mfcc_array.T)
    #         delta2_mfccs = librosa.feature.delta(mfcc_array.T, order=2)
    #         features_per_window[f'dmfcc{i}'] = delta_mfccs[0]
    #         features_per_window[f'ddmfcc{i}'] = delta2_mfccs[0]
    #     else:
    #         features_per_window[f'dmfcc{i}'] = [0]
    #         features_per_window[f'ddmfcc{i}'] = [0]
    
    # # Add jitter and shimmer to features_per_window
    # features_per_window['jitter'] = list(jitter) + [0]  # Add padding for the last window
    # features_per_window['shimmer'] = list(shimmer) + [0]  # Add padding for the last window
    
    # Return all features for all windows
    return features_per_window

# -------------------- 5. Procesare fisiere --------------------
def process_files(folder_path, output_path):
    data_list = []

    # Splitting the subjects
    # Retrieving the subjects
    files = os.listdir(folder_path)

    # Process train_subjects
    data_list = process_subjects(folder_path,
                                 files)
    
    # Crearea DataFrame-ului
    columns = ["ID"]
   
    # Feature names
    feature_names = [   'mfcc0',
                        'mfcc1',
                        'mfcc2',
                        'mfcc3',
                        'mfcc4',
                        'mfcc5',
                        'mfcc6',
                        'mfcc7',
                        'mfcc8',
                        'mfcc9',
                        'mfcc10',
                        'mfcc11',
                        'mfcc12',
                        'fundamental_freq',
                        'ttp'
                        ]
    
    columns.extend([f"{f}" for f in feature_names])

    # Save the csv
    df = pd.DataFrame(data_list, columns=columns)
    os.makedirs(output_path, exist_ok=True)
    output_csv = os.path.join(output_path, f"{FS}_{WS}_{OVR}_{WINDOW}_{FILT}_MFCCS_F0.csv")
    df.to_csv(output_csv, index=False)
    print(f"Datele au fost salvate în {output_csv}")


def process_subject(audio):
    # Load data
    # _, data = wav.read(file_name)

    # Median filtering
    if FILT == "median":
        audio = median_filter_1d(audio, kernel_size=7)

    # Min-max normalization
    audio = audio / np.max(np.abs(audio))

    # Windowing
    windows = create_windows(audio)
    return windows


def process_subjects(folder_path, files):
    data_list = []
    
    for file in files:
        if file.endswith('.wav'):  # Assuming you're processing WAV files
            file_path = os.path.join(folder_path, file)
            
            # Extract ID from filename
            file_id = file.split('.')[0]  # Adjust this based on your naming convention
            
            # Load audio and extract windows
            audio, _ = librosa.load(file_path, sr=FS)
            windows = process_subject(audio)  # You'll need to implement this function
            
            # Calculate features
            features = calculate_features(windows)
            
            # Append ID and features to data_list
            features = calculate_features(windows)
            num_windows = len(features['ttp'])  # Use any feature to get the window count

            # For each window, create a row with file_id and all features for that window
            for window_idx in range(num_windows):
                row = [file_id]
                for key in features:
                    row.append(features[key][window_idx])
                data_list.append(row)
    
    return data_list


# def plot_spectrum(freq, spectrum, xlabel='Frecv', ylabel='Magn', filename_prefix='test'):
#     plt.figure(figsize=(15, 10))

#     # Plot the single spectrum
#     plt.stem(freq, spectrum, linefmt=f'b-', markerfmt=f'bx', basefmt=" ", label='Signal Spectrum')
#     plt.title("Single Channel Spectrum")
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.tight_layout()

#     # Save the plot to a file
#     plt.legend(loc='upper right', ncol=2, fontsize=8)
#     plt.tight_layout()
#     plt.savefig(f'{filename_prefix}_spectrum.png')
#     plt.close()


# def save_channels_plot(t, filtered_data, x_label="Timp (s)", y_label='Amplitudine', title=''):
#     # Grafic înainte de filtrare NLMS
#     plt.figure(figsize=(15, 10))
#     for i in range(NUM_CHANNELS):
#         plt.subplot(NUM_CHANNELS, 1, i + 1)
#         plt.plot(t, filtered_data[i])
#         plt.title(f"CH {i+1}")
#         plt.xlabel(x_label)
#         plt.ylabel(y_label)
#         plt.tight_layout()
#     plt.savefig(f'{title}.png')
#     plt.close()


# # Plot spectrums on the same plot using stem and valid colors
# def plot_spectrums_stem(freq, spectrum1, spectrum2, xlabel, ylabel, filename_prefix):
#     plt.figure(figsize=(15, NUM_CHANNELS))

#     # Plot spectrum1 (before filtering)
#     plt.figure(figsize=(15, 10))
#     for i in range(NUM_CHANNELS):
#         plt.subplot(NUM_CHANNELS, 1, i + 1)
#         plt.stem(freq, spectrum1[i], linefmt=f'b-', markerfmt=f'bx', basefmt=" ", label=f'Canal {i+1} - Before')
#         plt.title(f"CH {i+1}")
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         plt.tight_layout()


#     # Plot spectrum2 (after filtering)
#     for i in range(spectrum2.shape[0]):
#         plt.subplot(NUM_CHANNELS, 1, i + 1)
#         plt.stem(freq, spectrum2[i], linefmt=f'r--', markerfmt=f'rx', basefmt=" ", label=f'Canal {i+1} - Before')
#         plt.title(f"CH {i+1}")
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         plt.tight_layout()
        

#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title("Compararea Spectrelor (Before vs After NLMS)")
#     plt.legend(loc='upper right', ncol=2, fontsize=8)
#     plt.tight_layout()
#     plt.savefig(f'{filename_prefix}_comparison_spectrums.png')
#     plt.close()


# def exponential_moving_average(signal, alpha=0.1):
#     ema_signal = np.zeros_like(signal)
#     ema_signal[0] = signal[0]
#     for i in range(1, len(signal)):
#         ema_signal[i] = alpha * signal[i] + (1 - alpha) * ema_signal[i-1]
#     return ema_signal


# -------------------- 6. Salvare DataFrame --------------------
if __name__ == "__main__":
    folder_path = "database/timit"  
    output_path = "database/features"  

    if ALL_TRAIN:
        CROSS_VAL_SPLIT = False

    print("Started processing...")
    start = time()
    process_files(folder_path, output_path)
    end = time()
    print(f"Total processing time: {(end - start) / 60.} minutes.")