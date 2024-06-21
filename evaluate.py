from model import WhisperSegmenterFast, WhisperSegmenter
import librosa
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from datautils import get_audio_and_label_paths, read_label
import os

def evaluate( audio_list, label_list, segmenter, batch_size, max_length, num_trials, num_beams=4, target_cluster = None ):

    total_n_true_positive_segment_wise, total_n_positive_in_prediction_segment_wise, total_n_positive_in_label_segment_wise = 0,0,0
    total_n_true_positive_frame_wise, total_n_positive_in_prediction_frame_wise, total_n_positive_in_label_frame_wise = 0,0,0
    
    for audio, label in tqdm(zip(audio_list, label_list), total = len(audio_list)):        
        prediction = segmenter.segment(  audio, sr = label["sr"],
                       min_frequency = label.get("min_frequency", None),
                       spec_time_step = label.get("spec_time_step", None),
                                       
                       max_length = max_length, 
                       batch_size = batch_size, 
                       num_trials = num_trials,
                       num_beams = num_beams
                 )

        TP, P_pred, P_label = segmenter.segment_score( prediction, label,  target_cluster = target_cluster )[:3]
        total_n_true_positive_segment_wise += TP
        total_n_positive_in_prediction_segment_wise += P_pred
        total_n_positive_in_label_segment_wise += P_label
        
        TP, P_pred, P_label = segmenter.frame_score( prediction, label,  target_cluster = target_cluster )[:3]
        
        total_n_true_positive_frame_wise += TP
        total_n_positive_in_prediction_frame_wise += P_pred
        total_n_positive_in_label_frame_wise += P_label
        
    res = {}
    
    precision = total_n_true_positive_segment_wise / max(total_n_positive_in_prediction_segment_wise, 1e-12)
    recall = total_n_true_positive_segment_wise / max( total_n_positive_in_label_segment_wise, 1e-12 )
    f1 = 2/(1/max(precision, 1e-12) + 1/max(recall, 1e-12)  )
    
    res["segment_wise"] = [ total_n_true_positive_segment_wise, total_n_positive_in_prediction_segment_wise, total_n_positive_in_label_segment_wise, precision, recall, f1 ]
    
    
    precision = total_n_true_positive_frame_wise / max(total_n_positive_in_prediction_frame_wise, 1e-12)
    recall = total_n_true_positive_frame_wise / max( total_n_positive_in_label_frame_wise, 1e-12 )
    f1 = 2/(1/max(precision, 1e-12) + 1/max(recall, 1e-12)  )
    
    res["frame_wise"] = [ total_n_true_positive_frame_wise, total_n_positive_in_prediction_frame_wise, total_n_positive_in_label_frame_wise, precision, recall, f1 ]
    
    return res

def evaluate_dataset( dataset_folder, model_path, num_trials, max_length = 448, num_beams = 4, batch_size = 8, **kwargs ):
    audio_list, label_list = [], []
    audio_paths, label_paths = get_audio_and_label_paths(dataset_folder)
    for audio_path, label_path in zip(audio_paths, label_paths):
        label = read_label( label_path )
        audio, sr = librosa.load( audio_path, sr = label.get("sr", None) )
        label["sr"] = sr
        audio_list.append(audio)
        label_list.append(label) 
    try:
        segmenter = WhisperSegmenterFast(  model_path = model_path,  device = "cuda")
    except:
        segmenter = WhisperSegmenter(  model_path = model_path,  device = "cuda")
    res = evaluate( audio_list, label_list, segmenter, batch_size, max_length, num_trials, num_beams, target_cluster = None )

    all_res = {
        "segment_wise_scores": {"N-true-positive": res["segment_wise"][0],
                                "N-positive-in-prediction": res["segment_wise"][1],
                                "N-positive-in-ground-truth": res["segment_wise"][2],
                                "precision": res["segment_wise"][3],
                                "recall": res["segment_wise"][4],
                                "F1": res["segment_wise"][5]
                                },
        "frame_wise_scores": {"N-true-positive": res["frame_wise"][0],
                                "N-positive-in-prediction": res["frame_wise"][1],
                                "N-positive-in-ground-truth": res["frame_wise"][2],
                                "precision": res["frame_wise"][3],
                                "recall": res["frame_wise"][4],
                                "F1": res["frame_wise"][5]
                                }
    }
    return all_res