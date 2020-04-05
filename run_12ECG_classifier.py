import scipy.signal
import numpy as np
import torch

import math
import model

MODEL_CLASSES = ['AF', 'I-AVB', 'LBBB', 'PAC', 'PVC', 'RBBB', 'STD', 'STE']
MODEL_FOLDER = 'models/'


def filter_signal(ts, rate, low_freq=None, high_freq=None, order=4):
    if low_freq:
        lb_n_freq = low_freq / (rate/2)
        b, a = scipy.signal.butter(order, lb_n_freq, 'high')
        ts = scipy.signal.filtfilt(b, a, ts)
    if high_freq:
        hb_n_freq = high_freq / (rate/2)
        b, a = scipy.signal.butter(order, hb_n_freq, 'low')
        ts = scipy.signal.filtfilt(b, a, ts)
    ts = ts.copy()
    return ts

def process_ecg(ecg, divide_by=200, max_length=72000, low_pass_freq=40, downsample=5, trend_window=None):
    processed = np.zeros((ecg.shape[0], max_length // downsample), dtype='float32')    
    if ecg.shape[1] > max_length:
        ecg = ecg[:, -max_length:]
    for i in range(ecg.shape[0]):
        sig = ecg[i].astype('float32')
        sig = sig - np.median(sig)
        sig /= divide_by
        sig = filter_signal(sig, 500, low_freq=None, high_freq=low_pass_freq, order=4)
        sig = sig[::downsample]
        processed[i, -sig.shape[0]:] = sig
    return processed

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def run_12ECG_classifier(data, header_data, classes, models):
    classes = list(classes)
    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)
    
    processed_data = process_ecg(data)
    torch_data = torch.from_numpy(processed_data[None, :, :])
    
    for class_index, class_name in enumerate(MODEL_CLASSES):
        probs = list()
        preds = list()
        for fold in range(4):
            net = models[(fold, class_name)]
            net.eval()
            logit = net(torch_data).detach().cpu().numpy()[0, class_index]
            probability = sigmoid(logit)
            prediction = probability > 0.5
            
            probs.append(probability)
            preds.append(prediction)
            
        probability = np.mean(probs)
        prediction = np.sum(preds) > 1.0
        
        current_score[classes.index(class_name)] = probability
        current_label[classes.index(class_name)] = prediction
        
    if current_label.sum() == 0:
        current_label[classes.index('Normal')] = 1
        current_score[classes.index('Normal')] = 0.95
    else:
        current_label[classes.index('Normal')] = 0
        current_score[classes.index('Normal')] = 0.05        
        
    return current_label, current_score

def load_12ECG_model():
    models = dict()
    for fold in range(4):
        for class_name in MODEL_CLASSES:
            state_dict = torch.load(f'{MODEL_FOLDER}submit1_{class_name}_{fold}.pt')
            net = model.PhyChal2020Net()
            net.load_state_dict(state_dict)
            models[(fold, class_name)] = net
    return models