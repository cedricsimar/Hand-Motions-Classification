


from settings import Settings
import numpy as np

class BatchGenerator:

    """
    Generate batches to feed the RCNN
    """

    def __init__(self, file_path):
                
        # to adapt for the 3Gb file

        # load subjects data file(s)
        self.data, self.labels = np.load(file_path)

        # separate training from validation data
        # training data is of shape 6 x 119496 x 32 [series][signal][channel]
        self.training_data = self.data[Settings.SERIES_TRAINING_INDICES]
        self.training_labels = self.labels[Settings.SERIES_TRAINING_INDICES]

        self.validation_data = self.data[Settings.SERIES_VALIDATION_INDICES]
        self.validation_labels = self.labels[Settings.SERIES_VALIDATION_INDICES]

        # get indices of positive and negative data indices in each series
        self.positive_training_indices = [np.where(np.sum(self.training_labels[series], axis=1) > 0)[0] for series in range(len(self.training_data))]
        self.negative_training_indices = [np.where(np.sum(self.training_labels[series], axis=1) == 0)[0] for series in range(len(self.training_data))]


    def get_random_series(self, x):
        return(np.random.randint(len(x)))

    
    def get_random_point(self, series, x):
        return(np.random.randint(Settings.SAMPLE_LENGTH - 1, len(x[series])))
        

    def random_batch(self, purpose, size):

        """
        Generate a batch of fixed size for the training or validation
        We also generate a batch for validation since the number of sample doesn't fit in RAM
        """
        
        batch_data = []
        batch_labels = []

        if purpose == "training":
            x = self.training_data
            y = self.training_labels
        else:
            x = self.validation_data
            y = self.validation_labels

        for _ in range(size):

            series = self.get_random_series(x)
            point = self.get_random_point(series, x)   # time

            # sample is of dimension Settings.SAMPLE_LENGTH x 32
            sample = np.copy(x[series][point - Settings.SAMPLE_LENGTH + 1:point + 1, :])
            
            # per sample per channel mean substraction
            sample -= np.mean(sample, axis=0).reshape((1, Settings.NUM_CHANNELS))

            # reshape sample to 32 x Settings.SAMPLE_LENGTH
            sample = sample.T

            batch_data.append(sample)
            batch_labels.append(y[series][point])

        return((np.asarray(batch_data), np.asarray(batch_labels)))
        

    
