
import numpy as np
from scipy import signal

SERIES_TRAINING_INDICES = [0, 1, 3, 4, 6, 7]
SERIES_VALIDATION_INDICES = [2, 5]
FEAT_EXTR_STRIDE = 12

SAMPLE_LENGTH = 3584
SAMPLING_FREQUENCY = 500
WELCH_OVERLAP = 0.25

#                             delta   theta    alpha     beta    low-gamma     
PSD_FREQUENCIES = np.array([[0.1, 4], [4, 8], [8, 15], [15, 30], [30, 45]])

class FeaturesExtractor:

    def __init__(self, aggregated_data_path):

        # load subject aggregated data file
        self.data, self.labels = np.load(aggregated_data_path)
        
    
    def extract(self):

        self.raw_training_data = self.data[SERIES_TRAINING_INDICES]
        self.raw_training_labels = self.labels[SERIES_TRAINING_INDICES]

        self.training_set = []
        self.training_labels = []

        self.raw_validation_data = self.data[SERIES_VALIDATION_INDICES]
        self.raw_validation_labels = self.labels[SERIES_VALIDATION_INDICES]

        self.validation_set = []
        self.validation_labels = []

        # Build up and save the training and validation sets 
        
        for data, labels, dataset, labelset in [[self.raw_training_data, self.raw_training_labels, self.training_set, self.training_labels],
                                                [self.raw_validation_data, self.raw_validation_labels, self.validation_set, self.validation_labels]]:
        
            # for each series
            for series in range(len(data)):

                print("Preprocessing series", series + 1)

                start = 0
                end = SAMPLE_LENGTH
                
                # preprocess the raw data to extract the PSD feature
                while(end < len(data[series])):
                    
                    raw_data_chunk = data[series][start:end]  # shape (3584, 32)
                    
                    chunk_psd = self.compute_power_spectral_density(raw_data_chunk.T)
                    chunk_psd = np.ndarray.flatten(chunk_psd) # (5, 32) -> (1, 160)
                    
                    dataset.append(chunk_psd)
                    labelset.append(labels[series][end])

                    start += FEAT_EXTR_STRIDE
                    end += FEAT_EXTR_STRIDE
        
        # save the preprocessed training set and validation set
        np.save("../../../data/train/preprocessed/non_norm_psd_training_set.npy", [self.training_set, self.training_labels])
        np.save("../../../data/train/preprocessed/non_norm_psd_validation_set.npy", [self.validation_set, self.validation_labels])
        

    
    def compute_power_spectral_density(self, windowed_signal):
        
        """
        Compute the PSD of each 32 electrodes and form a binned spectrogram of 5 frequency bands
        Return the log_10 on the 32 spectrograms normalized by their total power
        """

        # Windowed signal of shape (32 x 3584)
        ret = []
        
        # Welch parameters
        sliding_window = SAMPLING_FREQUENCY
        overlap = WELCH_OVERLAP
        n_overlap = int(sliding_window * overlap)
        
        # compute psd using Welch method
        freqs, power = signal.welch(windowed_signal, fs=SAMPLING_FREQUENCY,
                                    nperseg=sliding_window, noverlap=n_overlap)
        
        for psd_freq in PSD_FREQUENCIES:
            tmp = (freqs >= psd_freq[0]) & (freqs < psd_freq[1])
            ret.append(power[:,tmp].mean(1))
        
        return(np.log(np.array(ret)))