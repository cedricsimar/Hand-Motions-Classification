
# Because notebooks are for bioinformaticians 
"""
        _
    .__(.)<  (KWA)
     \___)    

"""
"""
Dataset Nature paper: https://www.nature.com/articles/sdata201447

Observation on the dataset:
    - 96 451 / 119 496 are no movement (all 0)
    - 5 100 frames with motion events (1) in each column: 
        * 1 mark the begining of an event 
        * not the whole duration of the event, only 150 frames / 0.3s
        * the begining of the event is centered in the 150 frames
          (75 frames before and 75 frames after)
        * possible overlaps between 2 events 
        * possible periods of "no event" between 2 events of a same trial
"""

from settings import Settings
from dataAggregator import DataAggregator
from featuresExtractor import FeaturesExtractor
from batchGenerator import BatchGenerator

import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

def main(): 

    # initialize Settings class with json file
    Settings.load("./settings.json")

    # aggregate data per patient
    # DataAggregator().aggregate(Settings.TRAIN_DATA_CSV_PATH, Settings.TRAIN_AGG_DATA_PATH, purpose="train")
    
    # extract features from the raw aggregated data
    FeaturesExtractor(Settings.TRAIN_AGG_DATA_PATH + "sub1_aggregated_train.npy").extract()
    
    # load the preprocessed data
    training_set = np.load("../../../data/train/preprocessed/non_norm_psd_training_set.npy")
    validation_set = np.load("../../../data/train/preprocessed/non_norm_psd_validation_set.npy")
    
    # asarray(list(x)) to force the shape of x to be (77813, 160) and not (77813, )
    training_data = np.asarray(list(training_set[0]))
    training_labels = np.asarray(list(training_set[1]))

    validation_data = np.asarray(list(validation_set[0]))
    validation_labels = np.asarray(list(validation_set[1]))

    # from one-hot encoding to

    # balance the training set
    positive_training_indices = np.where(np.sum(training_labels, axis=1) > 0)[0] 
    negative_training_indices = np.where(np.sum(training_labels, axis=1) == 0)[0]

    np.random.shuffle(negative_training_indices) # in place
    
    training_data = np.concatenate((training_data[positive_training_indices], training_data[negative_training_indices[:len(positive_training_indices)]]))
    training_labels = np.concatenate((training_labels[positive_training_indices], training_labels[negative_training_indices[:len(positive_training_indices)]]))

    # clf = XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=6, subsample=0.50, 
    #                     colsample_bytree=0.50, colsample_bylevel=1.00, min_child_weight=2, seed=42)
    # clf = RandomForestClassifier(n_estimators=500, criterion="entropy")
    # clf = SVC(C=1.0, probability=True)

    clf = LogisticRegression(C=0.1) # 0.75 AUC

    predicted = np.empty((validation_data.shape[0], 6))
    for i in range(6):
        print("Fitting class: ", i+1)
        clf.fit(training_data, training_labels[:, i])
        predicted[:, i] = clf.predict_proba(validation_data)[:, 1]
        print("Series ROC AUC :", roc_auc_score(validation_labels[:, i], predicted[:, i]), "\n")


    print("Total ROC AUC: o//", roc_auc_score(validation_labels.reshape(-1), predicted.reshape(-1)), "\\\\o \n\n")
    
    assert(0) # <3


    


if __name__ == "__main__":
    main()