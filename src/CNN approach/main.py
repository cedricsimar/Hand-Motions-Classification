
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
from batchGenerator import BatchGenerator

import tensorflow as tf
from cnn import CNN

from sklearn.metrics import roc_auc_score

def main(): 

    # initialize Settings class with json file
    Settings.load("./settings.json")

    # aggregate data per patient
    # DataAggregator().aggregate(Settings.TRAIN_DATA_CSV_PATH, Settings.TRAIN_AGG_DATA_PATH, purpose="train")
    
    # initialize a batch generator using the aggregated data
    batchGen = BatchGenerator(Settings.TRAIN_AGG_DATA_PATH + "sub1_aggregated_train.npy")

    # initialize training session variables

    input_ph = tf.placeholder(tf.float32, [None, Settings.NUM_CHANNELS, Settings.SAMPLE_LENGTH])
    labels_ph = tf.placeholder(tf.float32, [None, Settings.NUM_EVENTS])
    
    nn = CNN(input_ph, labels_ph)

    tf_session = tf.Session()
    tf_session.run(tf.global_variables_initializer())

    # run training session loop
    for training_iteration in range(1000):
        
        # get next training batch
        batch_x, batch_y = batchGen.random_batch("training", Settings.MINIBATCH_SIZE)

        # run a training step
        tf_session.run(nn.optimize, {input_ph: batch_x, labels_ph: batch_y})

        # check the accuracy evolution every x steps
        if training_iteration % 10 == 0:

            # generate a validation batch 
            validation_x, validation_y = batchGen.random_batch("validation", 500)
            
            print("Accuracy:", tf_session.run(nn.predict_proba, {input_ph: validation_x,
                                               labels_ph: validation_y}))

            predicted = tf_session.run(nn.predict_proba, {input_ph: validation_x,
                                               labels_ph: validation_y})

            print("ROC AUC:", roc_auc_score(validation_y.reshape(-1), predicted.reshape(-1)))
            

    


if __name__ == "__main__":
    main()