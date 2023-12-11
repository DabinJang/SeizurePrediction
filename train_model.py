import os
import pickle
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import natsort
import tensorflow as tf
from tensorflow import keras
from keras import layers, losses
from keras.layers import Input
from keras.models import Model
from keras.metrics import CategoricalAccuracy, Recall, Precision
from keras.backend import clear_session
from datetime import datetime, timedelta
from pytz import timezone
import matplotlib.pyplot as plt
# private module

from dilationmodel import dilationnet
from dataload import load_dataset, get_used_state, psm_train_test_split,  interval_to_segment, segment_to_data, DataGenerator, DatasetHeader
import envset

WINDOW_SIZE = 2
OVERLAP_1SEC = 1
NON_OVERLAP = WINDOW_SIZE
SAMPLING_FREQUENCY = 128
CHECKPOINT_DIR = "./training_checkpoint/"

# None means we do not use that state.
# Number means a class of that state.

state_label = {
         'interictal': 0,
     'preictal_early': None,
    'preictal_ontime': 1,
      'preictal_late': 1,
              'ictal': None,
          'postictal': None,
             'unused': None,
             }

states_need_overlap = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early']
states_no_overlap = ['postictal', 'interictal', 'unused']

dataset_name = ['snu', 'chb', 'chb_psm']

dataset_header = {
"chb": DatasetHeader(
    name='chb',
    dir="/home/dataset/CHB",
    patient_info="./patient_info_chb.csv",
    train_info="./patient_info_chb_train.csv",
    test_info="./patient_info_chb_test.csv"
    ),
"snu": DatasetHeader(
    name='snu',
    dir="/home/dataset/SNU",
    patient_info="./patient_info_snu.csv",
    train_info="./patient_info_snu_train.csv",
    test_info="./patient_info_snu_test.csv"
    ),
"chb_psm": DatasetHeader(
    name='chb_psm',
    dir="/home/dataset/CHB",
    patient_info="./patient_info_chb_unused.csv"
    )
}

dataset_channels = {
    "snu": [
        'Fp1-AVG',
        # 'Fp2-AVG',
        # 'T3-AVG', 
        # 'T4-AVG',
        # 'O1-AVG',
        # 'O2-AVG'
    ],
    "chb": [
        'FP1-F7',
        'FP2-F8',
        'T7-P7',
        'T8-P8',
        'P7-O1',
        'P4-O2'
    ],
    "chb_psm":[
        'FP2-F8',
    ]
}

def calculate_fpr(tp, fp, tn, fn,window_size):
    return fp/(tp+tn)

def calculate_confusion_matrix_from_metrics(accuracy, precision, recall):
    # 정확도에 대한 혼동 행렬 비율 계산
    fn_fp = 1 - accuracy
    
    # 정밀도에 대한 혼동 행렬 비율 계산
    fp_over_tp = (1 / precision) - 1
    
    # 리콜에 대한 혼동 행렬 비율 계산
    fn_over_tp = (1 / recall) - 1
    
    # 혼동 행렬의 나머지 부분 계산
    tp = fn_fp/(fp_over_tp+fn_over_tp)
    fn = tp * fn_over_tp
    fp = tp * fp_over_tp
    tn = 1 - (tp + fn + fp)
    
    return [tp, fp, tn, fn]

def create_model(input_shape)->Model:
    inputs = Input(shape=input_shape)
    outputs = dilationnet(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_model(patient, channel):
    used_channels = channel
    target_patient = patient
    n_splits = 3 # number of k fold
    epochs = 100
    batch_size = 64
    target_dataset = 'chb_psm'
    header = dataset_header[target_dataset]
    patient_data = load_dataset(header.patient_info)
    
    if 'psm' in target_dataset:
        checkpoint_name = f"training_{target_dataset}/{target_patient}/{used_channels}"
        is_target_patient = [target_patient in name for name in patient_data['name']]
        patient_data = patient_data[is_target_patient]
    else:
        checkpoint_name = f"training_{target_dataset}"
    
    patient_data = get_used_state(patient_data, state_label)
    
    train_interval, test_interval = psm_train_test_split(patient_data)
    
    train_segment_list = list()
    test_segment_list = list()

    proportion = dict()
    for idx, state_belong_train_interval in enumerate(list(train_interval['state'].unique())):
        if state_belong_train_interval in states_need_overlap:
            sliding_size = OVERLAP_1SEC
        elif state_belong_train_interval in states_no_overlap:
            sliding_size = NON_OVERLAP
        else:
            sliding_size = NON_OVERLAP
        
        if state_belong_train_interval in ['preictal_ontime']:
            proportion[idx] = 0.36
        elif state_belong_train_interval in ['preictal_late']:
            proportion[idx] = 0.03
        else:
            proportion[idx] = 0.6
        
        columns = list(patient_data.columns)

        is_state_match = train_interval['state']==state_belong_train_interval        
        current_state = train_interval[is_state_match]
        interval_list = current_state[columns].values.tolist()
        train_segment = interval_to_segment(interval_list, header.dir, WINDOW_SIZE, sliding_size)

        is_state_match = test_interval['state']==state_belong_train_interval        
        current_state = test_interval[is_state_match]
        interval_list = current_state[columns].values.tolist()
        test_segment = interval_to_segment(interval_list, header.dir, WINDOW_SIZE, sliding_size)
        
        train_segment_list.append(np.array(train_segment))
        test_segment_list.append(np.array(test_segment))
        
    checkpoint_file = "cp.ckpt"
    # load|create model
    checkpoint_file_path = os.path.join(CHECKPOINT_DIR, checkpoint_name, checkpoint_file)
    
    if os.path.exists(checkpoint_file_path):
        return 0
        # print(f"Load model from '{checkpoint_file_path}'")
        # model = tf.keras.models.load_model(checkpoint_file_path)
    else:
        print(f"Create new model and save it on '{checkpoint_file_path}'")
        model = create_model((len(used_channels),WINDOW_SIZE*SAMPLING_FREQUENCY))
    
    model.compile(optimizer = 'Adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    metrics=[CategoricalAccuracy(),
                            Recall(class_id=0),
                            Precision(class_id=0),
                            ]
                    )
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file_path,
                                                    save_best_only=True,
                                                    verbose=1)
    
    val_loss_callback = envset.earlystop_callback(monitor="val_loss")
    
    train_generator = DataGenerator(data=train_segment_list,
                                    state_label=state_label,
                                    channels=used_channels,
                                    batch_size=batch_size,
                                    proportion=proportion)
    
    validation_generator = DataGenerator(data=test_segment_list,
                                            state_label=state_label,
                                            channels=used_channels, 
                                            batch_size=batch_size,
                                            proportion=proportion)
    history = model.fit(
                train_generator,
                epochs = epochs,
                validation_data = validation_generator,
                use_multiprocessing=True,
                workers=4,
                callbacks= [cp_callback,
                            val_loss_callback,
                            #envset.garbage_collection_callback()
                            ]
                )
    
    print(f'\n\n\ntrain done.\n\n\n')
    
    hsty = list(history.history.keys())
    val_accuracy = history.history[[x for x in hsty if 'val_categorical_accuracy' in x][0]][-1]
    val_precision = history.history[[x for x in hsty if 'val_precision' in x][0]][-1]
    val_recall = history.history[[x for x in hsty if 'val_recall' in x][0]][-1]
    
    
    try:            
        with open(f"{os.path.join(CHECKPOINT_DIR,checkpoint_name)}/trainHistoryDict", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
            print(f"save trainHistoryDict at {os.path.join(CHECKPOINT_DIR, checkpoint_name)}")
    except:
        print('SaveError: Cannot save history')
        
    try:
        cm = calculate_confusion_matrix_from_metrics(accuracy=val_accuracy,
                                                 precision=val_precision,
                                                 recall=val_recall)
        tp, fp, tn, fn = cm
        fpr = calculate_fpr(tp, fp, tn, fn, 2)
        result = {"tp":tp,"fp":fp,"tn":tn,"fn":fn,"fpr":fpr}
        with open(os.path.join(CHECKPOINT_DIR, checkpoint_name,'cm_fpr'),'wb') as fw:
            pickle.dump(result, fw)
    except:
        print('SaveError: Cannot save confusion matrix and fpr')   
        
if __name__=="__main__":
    envset.config_gpu()
    chb_patient = natsort.natsorted([x for x in os.listdir("/home/dataset/CHB") if x.startswith("CHB")])
    chb_channel = ['FP1-F7', 'FP2-F8', 'T7-P7', 'T8-P8', 'P7-O1', 'P4-O2']
    for channel in chb_channel:
        for patient in chb_patient:
            train_model(patient, [channel])
           