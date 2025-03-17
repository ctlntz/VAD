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
        all_features[f'dmfcc_{i}'] = []
        all_features[f'ddmfcc_{i}'] = []
    all_features.update({
        "fundamental_freq": [],
        "mean_abs_value": [],
        "zero_crossing_rate": [],
        "slope_sign_changes": [],
        "skewness": [],
        "rms": [],
        "mnf": [],
        "mdf": [],
        "ttp": [],
        "s2": [],
        "s3": [],
        "s4": [],
        "jitter": [],
        "shimmer": []
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
    """
    Apply a 1D median filter to a signal.
    
    Args:
        signal: Input 1D array
        kernel_size: Size of the median filter window (must be odd)
        
    Returns:
        Filtered signal
    """
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
    period_values, amplitude_envelopes = [], []
    
    # Init features list
    all_features = get_feature_dict()

    mfccs_list = [[] for _ in range(NMFCC)]

    for window in windows:
        # Temporal features
        mean_abs_value = np.mean(np.abs(window))
        zero_crossing_rate = np.sum(np.abs(np.diff(window)) > 0.005)
        slope_sign_changes = sum(map(lambda x: (x >= 0).astype(float), (-np.diff(window, prepend=1)[1:-1]*np.diff(window)[1:])))
        skewness = skew(window)
        rms = np.sqrt(np.mean(window**2))

        # Extract pitch (F0) using FFT
        fft_result = np.fft.rfft(window, NFFT)
        magnitude = np.abs(fft_result)
        max_idx = np.argmax(magnitude[1:]) + 1
        fundamental_freq = max_idx / NFFT * FS

        # Plot the spectrum (TEST PURPOSES ONLY)
        plot_spectrum(np.fft.rfftfreq(NFFT)[1:], magnitude[1:])

        # # F0 extraction: (yaapt)
        # f_min = 75          # min. F0 [Hz]
        # f_max = 300         # max. F0 [Hz]
        # SignalObj = pBASIC.SignalObj(window, FS)
        # try:
        #     PitchObj = pYAAPT.yaapt(SignalObj, **{'frame_length':WS, # Durata cadrelor in ms
        #                             'tda_frame_length':WS,
        #                             'f0_min':f_min, 'f0_max':f_max}) # Limitele in frecv pt frecv fundamentala
        #     fundamental_freq_2 = PitchObj.samp_values
        # except Exception as ex:
        #     fundamental_freq_2 = 0

        # Save fundamental periods for jitter
        fundamental_period = 1 / fundamental_freq
        period_values.append(fundamental_period)
        
        # MFCCS
        mfccs = librosa.feature.mfcc(y=window, sr=FS, n_mfcc=NMFCC, 
                                    n_fft=int(WS/1000*FS))
        
        for i in range(NMFCC): 
            mfccs_list[i].append(mfccs[i])
        
        # Save max amplitude for shimmer
        amplitude_envelopes.append(np.max(np.abs(window)))

        # Spectral domain features:
        f, Pxx = welch(window, fs=FS, nperseg=len(window))
        
        # Mean Frequency (MNF) = ∑(f * PSD) / ∑(PSD)
        mnf = np.sum(f * Pxx) / np.sum(Pxx)
        
        # Median Frequency (MDF): frecvența unde suma PSD-ului este 50% din total
        cumulative_power = np.cumsum(Pxx)
        mdf = f[np.where(cumulative_power >= np.sum(Pxx) / 2)[0][0]]
        
        # Total Spectral Power (TTP)
        ttp = np.sum(Pxx)
        
        # Spectral Moments
        s2 = np.sqrt(np.sum((f - mnf) ** 2 * Pxx) / np.sum(Pxx))  # Momentul 2
        s3 = np.sum((f - mnf) ** 3 * Pxx) / (s2 ** 3 * np.sum(Pxx))   # Momentul 3
        s4 = np.sum((f - mnf) ** 4 * Pxx) / (s2 ** 4 * np.sum(Pxx))   # Momentul 4

        # Store values in lists
        all_features["mean_abs_value"].append(mean_abs_value)
        all_features["zero_crossing_rate"].append(zero_crossing_rate)
        all_features["slope_sign_changes"].append(slope_sign_changes)
        all_features["skewness"].append(skewness)
        all_features["rms"].append(rms)
        all_features["fundamental_freq"].append(fundamental_freq)
        for i in range(NMFCC):
            all_features[f"mfcc_{i}"].append(mfccs[i])
        all_features["rms"].append(rms)
        all_features["mnf"].append(mnf)
        all_features["mdf"].append(mdf)
        all_features["ttp"].append(ttp)
        all_features["s2"].append(s2)
        all_features["s3"].append(s3)
        all_features["s4"].append(s4)

    # Yaapt for F0 (just to check)
    # F0 extraction: (yaapt)
    f_min = 75          # min. F0 [Hz]
    f_max = 300         # max. F0 [Hz]
    SignalObj = pBASIC.SignalObj(windows, FS)
    try:
        PitchObj = pYAAPT.yaapt(SignalObj, **{'frame_length':WS, # Durata cadrelor in ms
                                'tda_frame_length':WS,
                                'f0_min':f_min, 'f0_max':f_max}) # Limitele in frecv pt frecv fundamentala
        fundamental_freq_2 = PitchObj.samp_values
    except Exception as ex:
        fundamental_freq_2 = 0
    frq_2_mean = np.mean(fundamental_freq_2)
    frq_2_std = np.std(fundamental_freq_2)

    # Compute jitter and shimmer
    jitter = np.abs(np.diff(period_values))
    all_features["jitter"] = jitter
    shimmer = np.abs(20 * np.log10(np.divide(amplitude_envelopes[1:], amplitude_envelopes[:-1])))
    all_features["shimmer"] = shimmer

    # Deltas 
    for i in range(NMFCC): 
        delta_mfccs = librosa.feature.delta(np.array(mfccs_list[i]).T)
        delta2_mfccs = librosa.feature.delta(np.array(mfccs_list[i]).T, order=2)

        all_features[f"dmfcc_{i}"].append(delta_mfccs)
        all_features[f"ddmfcc_{i}"].append(delta2_mfccs)

    # Calculate mean and std for each feature
    feature_means = [np.mean(all_features[key]) for key in all_features]
    feature_stds = [np.std(all_features[key]) for key in all_features]
    all_features = zip(feature_means, feature_stds)

    # Concatenate results into a final feature vector
    final_features = []
    for feat_mean, feat_std in all_features:
        final_features.extend([feat_mean, feat_std])
    return final_features


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
    columns = ["ID", "Class"]
   
    # Feature names
    feature_names = list(get_feature_dict().keys())
    for f in feature_names:
        columns.extend([f'{f}_mean', f'{f}_std'])

    # Save the csv
    df = pd.DataFrame(data_list, columns=columns)
    os.makedirs(output_path, exist_ok=True)
    output_csv = os.path.join(output_path, f"{FS}_{WS}_{OVR}_{WINDOW}_{FILT}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Datele au fost salvate în {output_csv}")


def process_subject(file_name):
    # Load data
    _, data = wav.read(file_name)

    # Median filtering
    if FILT == "median":
        data = median_filter_1d(data, kernel_size=7)

    # Min-max normalization
    data = data / np.max(np.abs(data))

    # Windowing
    windows = create_windows(data)
    return windows


def process_subjects(folder_path, subject_files):
    data_list = []
    for file_name in sorted(subject_files):
        if file_name.endswith(".wav"):
            # Extract info
            id, g, idx, cls = file_name.split("_")

            # Process the subject and extract windows
            subject_path = os.path.join(folder_path, file_name)
            windows = process_subject(subject_path)

            # Feature computation
            features = calculate_features(windows)
            data_list.append([id, g, idx, cls.split('.')[0]] + features)                

    # Perform feature normalization
    data_list = feature_normalization(data_list)

    return data_list


def plot_spectrum(freq, spectrum, xlabel='Frecv', ylabel='Magn', filename_prefix='test'):
    plt.figure(figsize=(15, 10))

    # Plot the single spectrum
    plt.stem(freq, spectrum, linefmt=f'b-', markerfmt=f'bx', basefmt=" ", label='Signal Spectrum')
    plt.title("Single Channel Spectrum")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    # Save the plot to a file
    plt.legend(loc='upper right', ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_spectrum.png')
    plt.close()


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


def exponential_moving_average(signal, alpha=0.1):
    ema_signal = np.zeros_like(signal)
    ema_signal[0] = signal[0]
    for i in range(1, len(signal)):
        ema_signal[i] = alpha * signal[i] + (1 - alpha) * ema_signal[i-1]
    return ema_signal


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