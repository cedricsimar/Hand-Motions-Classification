
import os
import pandas as pd
import numpy as np

from settings import Settings

"""
Aggregate the data from csv files of all subjects in one .npy file
"""

class DataAggregator:

    def aggregate(self, csv_data_path, destination_path, purpose = "train"):
        
        assert(purpose in["train", "test"])
        if(not os.path.exists(destination_path)): os.mkdir(destination_path)
        aggregated_file_name = "aggregated_" + purpose + ".npy"
        aggregated_file_path = os.path.join(destination_path, aggregated_file_name)

        if(os.path.exists(aggregated_file_path)):
            print("Aggregated data file already exists: overwrite? (y/n) : ", end='')
            if(input() == 'n'): return()
            
        aggregated_data = []
        aggregated_labels = []

        for subject_id in range(Settings.NUM_SUBJECTS):

            print("Aggregating subject", subject_id+1)

            subject_data = []
            subject_labels = []

            for serie_id in range(Settings.NUM_SERIES):

                # load and append the data into subject_data

                file_name = "subj" + str(subject_id + 1) + "_series" + str(serie_id + 1) + "_data.csv"
                file_path = os.path.join(csv_data_path, file_name)
                serie_data = pd.read_csv(file_path)

                electrodes_names = list(serie_data.columns[1:]) # discard the index column
                subject_data.append(np.array(serie_data[electrodes_names], dtype="float32"))

                if(purpose == "train"):
                        
                    # load and append the labels into subject_labels

                    file_name = "subj" + str(subject_id + 1) + "_series" + str(serie_id + 1) + "_events.csv"
                    file_path = os.path.join(csv_data_path, file_name)
                    serie_labels = pd.read_csv(file_path)

                    events_names = list(serie_labels.columns[1:]) # discard the index column
                    subject_labels.append(np.array(serie_labels[events_names], dtype="float32"))
            
                        
            aggregated_data.append(subject_data)
            aggregated_labels.append(subject_labels)
    

        # save the aggregated data and labels
        np.save(aggregated_file_path, [aggregated_data, aggregated_labels])








