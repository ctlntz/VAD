
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import skew
import matplotlib.pyplot as plt
from time import time
import random
from sklearn.model_selection import KFold
from scipy.signal import iirnotch, filtfilt
from scipy import signal
from sklearn.preprocessing import StandardScaler
from scipy.signal import welch


# Global variables
FS = 8000   # Frecventa de esantionare (Hz)
WS = 20  # Window size (miliseconds)
OVR = 0.5  # Window overlap
LOW = 20
HIGH = 3900
NFFT = 256
NORM = 'min-max' # min-max
WINDOW = 'hamming'
NUM_CHANNELS = 1
CROSS_VAL_SPLIT = True  ## Whether to split the dataset or not (False for cross-validation)
N_SPLITS = 5
ALL_TRAIN = True
FREQ_INCLUDED = False

# -------------------- 1. Filtrare Bandpass Butterworth --------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=LOW, highcut=HIGH, fs=FS, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=1)

# -------------------- 2. Filtrare NLMS --------------------
def nlms_filter(data, mu=0.01, noise_frequency=50, fs=512):
    # Creează o referință de zgomot sinusoidal pe 50Hz
    t = np.arange(data.shape[1]) / fs
    noise_ref = np.sin(2 * np.pi * noise_frequency * t)
    
    filtered_data = np.zeros_like(data)
    for channel in range(data.shape[0]):
        # NLMS pentru fiecare canal
        x = data[channel]
        y = np.zeros_like(x)
        w = 0
        for i in range(len(x)):
            e = x[i] - w * noise_ref[i]
            w += mu * e * noise_ref[i]
            y[i] = e
        filtered_data[channel] = y
    return filtered_data

# -------------------- 2. Filtrare Notch --------------------
def notch_filter(data, f0=50, Q=30):
    # Signal implementation
    # b, a = signal.iirnotch(f0, Q, FS)
    # data_filtered = signal.filtfilt(b, a, data, axis=1)

    # Manual segmentation
    f0 = 50   # Frequency to be notched (Hz)
    bw = 2    # Bandwidth of the notch (Hz)# Design the Butterworth stopband 
    low = (f0 - bw / 2) / (FS / 2)  # Lower cutoff frequency (normalized)
    high = (f0 + bw / 2) / (FS / 2) # Upper cutoff frequency (normalized) 
    b, a = butter(N=4, Wn=[low, high], btype='bandstop', output='ba')
    data_filtered = signal.filtfilt(b, a, data, axis=1)
    return data_filtered

# -------------------- 3. Normalizare --------------------
def normalize_data(signal, lower_perc=1, upper_perc=99, mode='z-score', ch_statistics=None):
    # Step 1: Compute the lower and upper percentiles
    # lower_bound = np.percentile(signal, lower_perc, axis=1)
    # upper_bound = np.percentile(signal, upper_perc, axis=1)

    # # Step 2: Clip the signal to the percentile range
    # for i, (low, up) in enumerate(zip(lower_bound, upper_bound)):
    #     signal[i, :] = np.clip(signal[i, :], low, up)

    # t = range(signal.shape[1])
    # # save_channels_plot(t, signal, title='After percentile')
    # plot_hist_cdf(signal[0], 'After percentile')

    if mode == 'min-max':
        # # Step 3: Apply Min-Max scaling to bring the signal to [0, 1]
        min_val = np.min(signal, axis=1)
        max_val = np.max(signal, axis=1)
        for i, (min, max) in enumerate(zip(min_val, max_val)):
            signal[i] = 2 * (signal[i] - min) / (max - min) - 1
    elif mode == 'z-score':
        # Z-score norm:
        # mean = np.mean(signal, axis=1)
        # std = np.std(signal, axis=1)
        for i, (m, s) in enumerate(ch_statistics):
            signal[i] = (signal[i] - m) / s

    # save_channels_plot(t, signal, title='After min-max')
    # plot_hist_cdf(signal[0], 'After min-max')

    return signal

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


def feature_normalization(data_list):
    # Convert the list to a numpy array
    data_list = np.array(data_list, dtype=object)

    # Separate attributes and features
    attributes = data_list[:, :5]  # First 5 columns (non-numeric attributes)
    features = data_list[:, 5:].astype(float)  # Numeric features (convert to float)

    # Split into train and validation sets based on "Split"
    train_indices = attributes[:, 4] == "Train"
    val_indices = attributes[:, 4] == "Val"

    train_features = features[train_indices]
    val_features = features[val_indices]

    # Normalize features
    scaler = StandardScaler()
    scaler.fit(train_features)  # Fit on train features only

    train_features_normalized = scaler.transform(train_features)
    # Combine attributes and normalized features
    data_list_normalized = np.hstack([attributes[train_indices], train_features_normalized])

    if val_features.shape[0] != 0:
        val_features_normalized = scaler.transform(val_features)
        val_data = np.hstack([attributes[val_indices], val_features_normalized])

        # Combine train and validation data
        data_list_normalized = np.vstack([data_list_normalized, val_data])
    return data_list_normalized


# -------------------- 4. Ferestruire --------------------
def create_windows(data, window_size=WINDOW, overlap=0.5, fs=FS, type='hamming'):
    step = int(window_size * (1 - overlap) * fs / 1000)
    window_length = int(window_size * fs / 1000)
    windows = []
    for start in range(0, data.shape[1] - window_length + 1, step):
        window = data[:, start:start + window_length]
        if type == 'hamming':
            hamming_window = np.hamming(window.shape[1])
            hamming_window_2d = np.tile(hamming_window, (data.shape[0], 1))
            windows.append(window * hamming_window_2d)
        elif type == 'rect':
            windows.append(window)
    return np.array(windows)

# -------------------- 5. Calculul trasaturilor --------------------
def calculate_features(window, freq=False):
    features = []
    for channel in window:
        # Temporal features
        mean_abs_value = np.mean(np.abs(channel))
        zero_crossing_rate = np.sum(np.abs(np.diff(channel)) > 0.005)
        slope_sign_changes = sum(map(lambda x: (x >= 0).astype(float), (-np.diff(channel, prepend=1)[1:-1]*np.diff(channel)[1:])))
        waveform_length = np.sum(np.abs(np.diff(channel)))
        skewness = skew(channel)
        rms = np.sqrt(np.mean(channel**2))
        hjorth_activity = np.mean((channel - np.mean(channel))**2)
        
        # Spectral domain features:
        f, Pxx = welch(channel, fs=FS, nperseg=len(channel))
        
        # Mean Frequency (MNF) = ∑(f * PSD) / ∑(PSD)
        mnf = np.sum(f * Pxx) / np.sum(Pxx)
        
        # Median Frequency (MDF): frecvența unde suma PSD-ului este 50% din total
        cumulative_power = np.cumsum(Pxx)
        mdf = f[np.where(cumulative_power >= np.sum(Pxx) / 2)[0][0]]
        
        # Total Spectral Power (TTP)
        ttp = np.sum(Pxx)
        
        # Spectral Moments
        s1 = np.sum(f * Pxx) / np.sum(Pxx)  # Momentul 1 (MNF)
        s2 = np.sum((f - mnf) ** 2 * Pxx) / np.sum(Pxx)  # Momentul 2
        s3 = np.sum((f - mnf) ** 3 * Pxx) / np.sum(Pxx)  # Momentul 3
        
        win_features_list = [
            mean_abs_value,
            zero_crossing_rate,
            slope_sign_changes,
            waveform_length,
            skewness,
            rms,
            hjorth_activity,
            mnf,
            mdf,
            ttp,
            s1,
            s2,
            s3
        ]
        # Append all channel features
        features.extend(win_features_list)
    return features

# -------------------- 5. Procesare fisiere --------------------
def process_files(folder_path, output_path, fs, window_size=WINDOW, overlap=0.5, low=LOW, high=HIGH, norm_mode='min-max'):
    data_list = []

    # Splitting the subjects
    # Rertieving the subjects
    files = os.listdir(folder_path)
    subject_names = np.unique(["_".join(file_name.split("_")[:2])for file_name in files])
    random.shuffle(subject_names)

    # Shuffle the subjects
    if CROSS_VAL_SPLIT:
        file_path = "subject_splits.txt"
        if not os.path.exists(file_path):  # Verificăm dacă fișierul exists
            # Create new splits
            splits = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=42).split(subject_names))

            # Save the splits
            with open(file_path, "w") as f:
                for train_idx, val_idx in splits:
                    train_subjects = subject_names[train_idx].tolist()
                    val_subjects = subject_names[val_idx].tolist()
                    f.write(f"Train: {train_subjects} Val: {val_subjects}\n")

            print(f"Split-urile au fost salvate în '{file_path}'.")
        else:
            # Citirea split-urilor din fișier
            splits = []
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:  # Citim split-urile
                    train_part, val_part = line.strip().split(" Val: ")
                    train_subjects = eval(train_part.replace("Train: ", ""))
                    val_subjects = eval(val_part)
                    splits.append((train_subjects, val_subjects))
    elif ALL_TRAIN:
        splits = [(list(range(len(subject_names))),  list())]
    else:
        splits = [(list(range(0, 22)),  list(range(22,len(subject_names))))]

    for split_idx, (train_split, val_split) in enumerate(splits):
        # Compute channel statistics 
        train_data = [[] for _ in range (NUM_CHANNELS)]
        for file in files:
            nume, prenume, clasa, _ = file.split("_")
            id_subiect = f"{nume}_{prenume}"

            # Process the train split
            if id_subiect in train_split:
                # Read the data
                data = np.load(os.path.join(folder_path, file))               

                 # Filtrare bandpass
                filtered_data = bandpass_filter(data, lowcut=low, highcut=high, fs=FS, order=4)

                # Notch
                filtered_data = notch_filter(filtered_data)

                # Append the data to the channel list
                for i in range(data.shape[0]):
                    train_data[i].extend(filtered_data[i])
        
        # Compute the statistics on the train data
        ch_statistics = []
        for channel in train_data:
            mean = np.mean(channel) 
            std = np.std(channel)
            ch_statistics.append((mean, std))  
        np.save("train_set_ch_stats.npy", ch_statistics)
        del train_data
        
        # Process train_subjects
        data_list = process_subjects(folder_path,
                                     files, 
                                     ch_statistics, 
                                     train_split, 
                                     window_size=window_size, 
                                     overlap=overlap, 
                                     low=low, 
                                     high=high, 
                                     norm_mode=norm_mode)
    
        # Crearea DataFrame-ului
        columns = ["ID", "Clasa", "Split"]
        
        # feature_names = ['MAV', 'ZCR', 'SSC', 'WL', 'SK', 'RMS', 'HA', 'ISEMG', 'MNF', 'MDF', 'TTP', 'MNP', 'PKF']
        feature_names = ['MAV', 'ZCR', 'SSC', 'WL', 'SK', 'RMS', 'HA', 'ISEMG', 'MNF', 'MDF', 'TTP', 'S1', 'S2', 'S3']
        for i in range(NUM_CHANNELS):
            columns.extend([f"Ch_{i+1}_{f}" for f in feature_names])

        # Save the csv
        df = pd.DataFrame(data_list, columns=columns)
        if ALL_TRAIN:
            output_csv = os.path.join(output_path, f"audio_all_train.csv")
        else:
            output_csv = os.path.join(output_path, f"audio_{FS}_{WS}_{OVR}_{WINDOW}_all_split_{split_idx}.csv")
        df.to_csv(output_csv, index=False)
        print(f"Datele au fost salvate în {output_csv}")
    return

def process_subject(file_name, ch_statistics, window_size=WINDOIW, overlap=0.5, low=LOW, high=HIGH, norm_mode='min-max'):
    # Încarcă datele
    data = np.load(file_name)

    # Eliminare canal 5
    keep_ch = [0, 1, 2, 3, 5, 6, 7]
    data = data[keep_ch, :]

    # Filtrare bandpass
    filtered_data = bandpass_filter(data, lowcut=low, highcut=high, fs=FS, order=4)
    
    # Filtrare NLMS
    # spectrum1 = np.abs(np.fft.rfft(filtered_data, n=NFFT, axis=1)) / NFFT
    # save_channels_plot(freq, spectrum1, 'f [Hz]', 'Abs', 'before')

    # filtered_data = nlms_filter(filtered_data, mu=0.01, noise_frequency=50, fs=FS)

    # Notch
    filtered_data = notch_filter(filtered_data)

    # spectrum2 = np.abs(np.fft.rfft(filtered_data, n=NFFT, axis=1)) / NFFT
    # save_channels_plot(freq, spectrum2, 'f [Hz]', 'Abs', 'after')
    # plot_spectrums_stem(freq, spectrum1, spectrum2, 'f [Hz]', 'Abs', 'overlap')

    # Plot inainte de normalizare
    # t = range(filtered_data.shape[1])
    # save_channels_plot(t, filtered_data, title='Before normalization')

    # plot_hist_cdf(filtered_data[0], title='Before normalization')

    ## Test alignment:
    # best_shift, filtered_data = align_circular_permutation(references[int(clasa)], filtered_data)
    # print(f"For {file_name}. Shift: {best_shift}.")

    # Z-score
    normalized_data = normalize_data(signal=filtered_data, mode=norm_mode, ch_statistics=ch_statistics)
    # save_channels_plot(t, normalized_data, 'Time', 'Amplitude', 'Valori normalizate')

    # Windowing
    windows = create_windows(normalized_data, window_size=window_size, overlap=overlap, fs=FS, type=WINDOW)
    return windows


def process_subjects(folder_path, subject_files, ch_statistics, train_split, window_size=250, overlap=0.5, low=20, high=250, norm_mode='z-score'):
    data_list = []
    for file_name in sorted(subject_files):
        if file_name.endswith(".npy"):
            # Extrage informații din numele fișierului
            parts = file_name.split("_")
            nume, prenume, clasa, _ = parts
            id_subject = f"{nume}_{prenume}"
            print(f"Processing subject: {nume}, exercise: {clasa}.")

            # Process the subject and extract windows
            subject_path = os.path.join(folder_path, file_name)
            windows = process_subject(subject_path, ch_statistics, window_size, overlap, low, high, norm_mode)

            # Calculul feature computation
            for window in windows:
                features = calculate_features(window, FREQ_INCLUDED)
                data_list.append([id_subject, nume, "", clasa, 'Train' if id_subject in train_split else 'Val'] + features)

    # Perform feature normalization
    data_list = feature_normalization(data_list)

    return data_list


def plot_hist_cdf(signal, title, low=1, up=99):
    p2, p98 = np.percentile(signal, low), np.percentile(signal, up)

    # Sortarea semnalului pentru funcția de repartiție (CDF)
    sorted_signal = np.sort(signal)
    cdf = np.linspace(0, 1, len(sorted_signal))

    # Creare subploturi
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 rând, 2 coloane

    # Subplot 1: Histograma
    axes[0].hist(signal, bins=60, color='blue', alpha=0.6, edgecolor='black', density=True)
    axes[0].axvline(x=p2, color='red', linestyle='--', linewidth=2, label=f'Percentile {low}%')
    axes[0].axvline(x=p98, color='red', linestyle='--', linewidth=2, label=f'Percentile {up}%')
    axes[0].set_title('Histogram', fontsize=16)
    axes[0].set_xlabel('Signal value', fontsize=14)
    axes[0].set_ylabel('Probability', fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(alpha=0.3)

    # Subplot 2: Funcția de Repartiție (CDF)
    axes[1].plot(sorted_signal, cdf, color='orange', linewidth=2, label='CDF')
    axes[1].axvline(x=p2, color='red', linestyle='--', linewidth=2, label=f'Percentile {low}%')
    axes[1].axvline(x=p98, color='red', linestyle='--', linewidth=2, label=f'Percentile {up}%')
    axes[1].set_title('Cumulative Distribution Function (CDF)', fontsize=16)
    axes[1].set_xlabel('Signal value', fontsize=14)
    axes[1].set_ylabel('Cumulative Probability', fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(alpha=0.3)

    # Configurare generală și salvare
    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Lasă spațiu pentru titlu
    plt.savefig(f'hist_cdf_{title}.png')
    plt.close()

def save_channels_plot(t, filtered_data, x_label="Timp (s)", y_label='Amplitudine', title=''):
    # Grafic înainte de filtrare NLMS
    plt.figure(figsize=(15, 10))
    for i in range(NUM_CHANNELS):
        plt.subplot(NUM_CHANNELS, 1, i + 1)
        plt.plot(t, filtered_data[i])
        plt.title(f"CH {i+1}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.close()


# Plot spectrums on the same plot using stem and valid colors
def plot_spectrums_stem(freq, spectrum1, spectrum2, xlabel, ylabel, filename_prefix):
    plt.figure(figsize=(15, NUM_CHANNELS))

    # Plot spectrum1 (before filtering)
    plt.figure(figsize=(15, 10))
    for i in range(NUM_CHANNELS):
        plt.subplot(NUM_CHANNELS, 1, i + 1)
        plt.stem(freq, spectrum1[i], linefmt=f'b-', markerfmt=f'bx', basefmt=" ", label=f'Canal {i+1} - Before')
        plt.title(f"CH {i+1}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()


    # Plot spectrum2 (after filtering)
    for i in range(spectrum2.shape[0]):
        plt.subplot(NUM_CHANNELS, 1, i + 1)
        plt.stem(freq, spectrum2[i], linefmt=f'r--', markerfmt=f'rx', basefmt=" ", label=f'Canal {i+1} - Before')
        plt.title(f"CH {i+1}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Compararea Spectrelor (Before vs After NLMS)")
    plt.legend(loc='upper right', ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_comparison_spectrums.png')
    plt.close()


def exponential_moving_average(signal, alpha=0.1):
    ema_signal = np.zeros_like(signal)
    ema_signal[0] = signal[0]
    for i in range(1, len(signal)):
        ema_signal[i] = alpha * signal[i] + (1 - alpha) * ema_signal[i-1]
    return ema_signal


# -------------------- 6. Salvare DataFrame --------------------
if __name__ == "__main__":
    folder_path = "db\\"  
    output_path = "db"  

    if ALL_TRAIN:
        CROSS_VAL_SPLIT = False

    print("Started processing...")
    start = time()
    process_files(folder_path, output_path, fs=FS, window_size=WS, overlap=OVR, low=LOW, high=HIGH, norm_mode=NORM)
    end = time()
    print(f"Total processing time: {(end - start) / 60.} minutes.")
