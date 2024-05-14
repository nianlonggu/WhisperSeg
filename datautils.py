import os,sys,inspect
import librosa
import pandas as pd
import numpy as np
import threading
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import json
from audio_utils import WhisperSegFeatureExtractor
from utils import RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP
from glob import glob

def random_mask(signal_length, mask_ratio):
    # Calculate the total number of points to mask
    total_mask_points = int(mask_ratio * signal_length)
    
    # Create an array of False (indicating unmasked)
    mask = np.zeros(signal_length, dtype=bool)
    
    # Randomly choose mask start points and lengths
    while total_mask_points > 0:
        start = np.random.randint(0, signal_length)
        # Randomly determine the length of this mask segment
        segment_length = min(np.random.randint(1, total_mask_points + 1), signal_length - start)
        mask[start:start + segment_length] = True
        total_mask_points -= segment_length
    
    return mask

# def get_audio_and_label_paths( folder ):
#     wav_list = [ folder + "/" + fname for fname in os.listdir( folder ) if fname.endswith(".wav") ]
#     audio_paths = []
#     label_paths = []
#     for wav_name in wav_list:
#         label_name = wav_name[:-4] + ".json"
#         if os.path.exists(label_name):
#             audio_paths.append( wav_name )
#             label_paths.append( label_name )
    
#     return audio_paths, label_paths

def get_audio_and_label_paths( folder ):
    label_fname_list = sorted( [ folder + "/" + fname for fname in os.listdir( folder ) if fname.endswith("_1.json") ] )
    all_paths = []
    for label_fname in label_fname_list:
        audio_base_fname = label_fname[ : -len("_1.json") ]
        audio_fname_list = sorted(glob( audio_base_fname + "*.wav" ), key = lambda x:int( x.split("_")[-1].split(".wav")[0] ) )
        all_paths.append(
            {
                "audios":audio_fname_list,
                "label":label_fname
            }
        )
    assert len(set([ len(item["audios"]) for item in all_paths ])) == 1
    audio_paths, label_paths = [], []
    for item in all_paths:
        audio_paths.append( item["audios"] )
        label_paths.append( item["label"] )
    return audio_paths, label_paths


def get_cluster_codebook( label_paths, initial_cluster_codebook ):
    cluster_codebook = deepcopy( initial_cluster_codebook )
    
    unique_clusters = []
    for label_file in label_paths:
        label = json.load( open(label_file) )
        unique_clusters += [ str(cluster) for cluster in label["cluster"]   ]
            
    unique_clusters = sorted(list(set(unique_clusters)))
    
    for cluster in unique_clusters:
        if cluster not in cluster_codebook:
            cluster_codebook[cluster] = len(cluster_codebook)
    return cluster_codebook

# def load_audio_and_label( audio_path_list, label_path_list, thread_id, audio_dict, label_dict, cluster_codebook ):
#     local_audio_list = []
#     local_label_list = []
    
#     for count, (audio_path, label_path) in enumerate(zip( audio_path_list, label_path_list )):
#         label = json.load(open( label_path ))
#         y, _ = librosa.load( audio_path, sr = label["sr"] )
                
#         local_audio_list.append( y )

#         onset_arr = np.array( label["onset"] )
#         offset_arr = np.array( label["offset"] )
#         valid_indices = np.logical_and( np.logical_and(  onset_arr < len(y)/label["sr"], offset_arr > 0 ),
#                                         onset_arr < offset_arr )
#         onset_arr = onset_arr[valid_indices]
#         offset_arr = offset_arr[valid_indices]
#         onset_arr[ onset_arr < 0 ] = 0
#         offset_arr[ offset_arr > len(y)/label["sr"] ] = len(y)/label["sr"]

#         label["cluster"] = [ label["cluster"][idx] for idx in np.argwhere(valid_indices)[:,0] ]        
#         cluster_id_arr = np.array( [ cluster_codebook[ value ] for value in label["cluster"] ]  )
        
#         label.update( {
#             "onset":onset_arr,
#             "offset":offset_arr,
#             "cluster_id":cluster_id_arr
#         } )
#         local_label_list.append( label )

#         if count % 10 == 0:
#             progress = count / len(audio_path_list)
#             print("|%s%s|progress: %.2f %%"%( "-" * (int( progress * 20 )), " "*( 20- int( progress * 20 )), progress*100 ), end = "\r", flush=True)
    
#     progress = 1.0
#     print("|%s%s|progress: %.2f %%"%( "-" * (int( progress * 20 )), " "*( 20- int( progress * 20 )), progress*100 ), end = "\r", flush=True)
    
#     audio_dict[thread_id] = local_audio_list
#     label_dict[thread_id] = local_label_list

def load_audio_and_label( audio_path_list, label_path_list, thread_id, audio_dict, label_dict, cluster_codebook ):
    local_audio_list = []
    local_label_list = []
    
    for count, (audio_paths, label_path) in enumerate(zip( audio_path_list, label_path_list )):        
        label = json.load(open( label_path ))
        audios = []
        for audio_path in audio_paths:
            y, _ = librosa.load( audio_path, sr = label["sr"] )
            audios.append(y)
        max_len = max( [ len(y) for y in audios ] )
        for idx in range(len(audios)):
            audios[idx] = np.concatenate( [ audios[idx], np.zeros( max_len - len(audios[idx] ) ) ], 
                                            axis = 0 ).astype(np.float32)
        ## from now on, y will be multi-channel audio
        y = np.asarray( audios )
        local_audio_list.append( y )

        onset_arr = np.array( label["onset"] )
        offset_arr = np.array( label["offset"] )
        valid_indices = np.logical_and( np.logical_and(  onset_arr < y.shape[1]/label["sr"], offset_arr > 0 ),
                                        onset_arr < offset_arr )
        onset_arr = onset_arr[valid_indices]
        offset_arr = offset_arr[valid_indices]
        onset_arr[ onset_arr < 0 ] = 0
        offset_arr[ offset_arr > y.shape[1]/label["sr"] ] = y.shape[1]/label["sr"]

        label["cluster"] = [ label["cluster"][idx] for idx in np.argwhere(valid_indices)[:,0] ]        
        cluster_id_arr = np.array( [ cluster_codebook[ value ] for value in label["cluster"] ]  )
        
        label.update( {
            "onset":onset_arr,
            "offset":offset_arr,
            "cluster_id":cluster_id_arr
        } )
        local_label_list.append( label )

        if count % 10 == 0:
            progress = count / len(audio_path_list)
            print("|%s%s|progress: %.2f %%"%( "-" * (int( progress * 20 )), " "*( 20- int( progress * 20 )), progress*100 ), end = "\r", flush=True)
    
    progress = 1.0
    print("|%s%s|progress: %.2f %%"%( "-" * (int( progress * 20 )), " "*( 20- int( progress * 20 )), progress*100 ), end = "\r", flush=True)
    
    audio_dict[thread_id] = local_audio_list
    label_dict[thread_id] = local_label_list

# def load_data(audio_path_list, label_path_list, cluster_codebook = None, n_threads = 1 ):
#     samples_per_thread = int(np.ceil( len(audio_path_list) / n_threads ))
#     audio_dict = {}
#     label_dict = {}
#     thread_list = []
    
#     for thread_id, offset in enumerate(range( 0, len(audio_path_list), samples_per_thread )):
#         t = threading.Thread( target=load_audio_and_label, args=( audio_path_list[offset:offset+samples_per_thread], 
#                                                           label_path_list[offset:offset+samples_per_thread],
#                                                           thread_id,
#                                                           audio_dict, label_dict,
#                                                           cluster_codebook
#                                                         ) )
#         t.start()
#         thread_list.append(t)
#     for t in thread_list:
#         t.join()
        
#     audio_list = []
#     label_list = []
#     for thread_id in sorted(audio_dict.keys()):
#         audio_list += audio_dict[thread_id]
#         label_list += label_dict[thread_id]
    
#     assert len(audio_list) == len(label_list) 
    
#     return audio_list, label_list

def load_data(audio_path_list, label_path_list, cluster_codebook = {}, n_threads = 1 ):
    samples_per_thread = int(np.ceil( len(audio_path_list) / n_threads ))
    audio_dict = {}
    label_dict = {}
    thread_list = []
    
    for thread_id, offset in enumerate(range( 0, len(audio_path_list), samples_per_thread )):
        t = threading.Thread( target=load_audio_and_label, args=( audio_path_list[offset:offset+samples_per_thread], 
                                                          label_path_list[offset:offset+samples_per_thread],
                                                          thread_id,
                                                          audio_dict, label_dict,
                                                          cluster_codebook
                                                        ) )
        t.start()
        thread_list.append(t)
    for t in thread_list:
        t.join()
        
    audio_list = []
    label_list = []
    for thread_id in sorted(audio_dict.keys()):
        audio_list += audio_dict[thread_id]
        label_list += label_dict[thread_id]
    
    assert len(audio_list) == len(label_list) 
    
    return audio_list, label_list

# def split_audio_and_label( audio, label, split_ratio ):
#     num_samples_in_audio = len(audio)
#     split_point = int( num_samples_in_audio * split_ratio )
#     split_time = split_point / label["sr"] 
    
#     audio_part1 = audio[ :split_point ]
#     intersected_indices_part1 = label["onset"] < split_time
#     label_part1 = deepcopy( label )
#     label_part1.update(
#     {
#         "onset":label["onset"][intersected_indices_part1],
#         "offset": np.minimum(label["offset"][intersected_indices_part1], split_time ),
#         "cluster_id":label["cluster_id"][intersected_indices_part1],
#         "cluster": [ label["cluster"][idx] for idx in np.argwhere( intersected_indices_part1 )[:,0]  ]
#     })
#     ## drop too short audios
#     if len(audio_part1) / label["sr"] < 0.1:
#         audio_part1 = None
#         label_part1 = None
    
    
#     audio_part2 = audio[ split_point: ]
#     intersected_indices_part2 = label["offset"] > split_time
#     label_part2 = deepcopy( label )
#     label_part2.update(
#     {
#         "onset": np.maximum(label["onset"][intersected_indices_part2], split_time ) - split_time,
#         "offset": label["offset"][intersected_indices_part2] - split_time,
#         "cluster_id":label["cluster_id"][intersected_indices_part2],
#         "cluster": [ label["cluster"][idx] for idx in np.argwhere( intersected_indices_part2 )[:,0] ]
#     })

#     ## drop too short audios
#     if len(audio_part2) / label["sr"] < 0.1:
#         audio_part2 = None
#         label_part2 = None
    
#     return ( audio_part1, label_part1 ), ( audio_part2, label_part2 )

def split_audio_and_label( audio, label, split_ratio ):
    num_samples_in_audio = audio.shape[1]
    split_point = int( num_samples_in_audio * split_ratio )
    split_time = split_point / label["sr"] 
    
    audio_part1 = audio[:, :split_point ]
    
    intersected_indices_part1 = label["onset"] < split_time
    label_part1 = deepcopy( label )
    label_part1.update(
    {
        "onset":label["onset"][intersected_indices_part1],
        "offset": np.minimum(label["offset"][intersected_indices_part1], split_time ),
        "cluster_id":label["cluster_id"][intersected_indices_part1],
        "cluster": [ label["cluster"][idx] for idx in np.argwhere( intersected_indices_part1 )[:,0]  ]
    })
    ## drop too short audios
    if audio_part1.shape[1] / label["sr"] < 0.1:
        audio_part1 = None
        label_part1 = None
        
    audio_part2 = audio[:, split_point: ]
    intersected_indices_part2 = label["offset"] > split_time
    label_part2 = deepcopy( label )
    label_part2.update(
    {
        "onset": np.maximum(label["onset"][intersected_indices_part2], split_time ) - split_time,
        "offset": label["offset"][intersected_indices_part2] - split_time,
        "cluster_id":label["cluster_id"][intersected_indices_part2],
        "cluster": [ label["cluster"][idx] for idx in np.argwhere( intersected_indices_part2 )[:,0] ]
    })

    ## drop too short audios
    if audio_part2.shape[1] / label["sr"] < 0.1:
        audio_part2 = None
        label_part2 = None
    
    return ( audio_part1, label_part1 ), ( audio_part2, label_part2 )

# def train_val_split( audio_list, label_list, val_ratio ):
    
#     audio_list_train = []
#     label_list_train = []
#     audio_list_val = []
#     label_list_val = []
    
#     for audio, label in zip( audio_list, label_list ):
#         mode = np.random.choice([0,1])
#         if mode == 0:
#             (audio_val, label_val), (audio_train, label_train) = split_audio_and_label( audio, label, val_ratio )
#         else:
#             (audio_train, label_train), (audio_val, label_val) = split_audio_and_label( audio, label, 1-val_ratio )
        
#         if audio_train is not None:
#             audio_list_train.append( audio_train )
#             label_list_train.append( label_train )
        
#         if audio_val is not None:
#             audio_list_val.append( audio_val )
#             label_list_val.append( label_val )
    
#     return (audio_list_train, label_list_train), ( audio_list_val, label_list_val )

def train_val_split( audio_list, label_list, val_ratio ):
    
    audio_list_train = []
    label_list_train = []
    audio_list_val = []
    label_list_val = []
    
    for audio, label in zip( audio_list, label_list ):
        mode = np.random.choice([0,1])
        if mode == 0:
            (audio_val, label_val), (audio_train, label_train) = split_audio_and_label( audio, label, val_ratio )
        else:
            (audio_train, label_train), (audio_val, label_val) = split_audio_and_label( audio, label, 1-val_ratio )
        
        if audio_train is not None:
            audio_list_train.append( audio_train )
            label_list_train.append( label_train )
        
        if audio_val is not None:
            audio_list_val.append( audio_val )
            label_list_val.append( label_val )
    
    return (audio_list_train, label_list_train), ( audio_list_val, label_list_val )

# def slice_audio_and_label( audio, label, total_spec_columns ):
#     sr = label["sr"]
#     clip_duration = total_spec_columns * label["spec_time_step"]
    
#     num_samples_in_clip = int( np.round( clip_duration * sr ) )
#     padded_audio = np.concatenate( [ np.zeros( num_samples_in_clip ), audio ], axis = 0 )
#     padded_label = {
#         "onset": label["onset"] + clip_duration,
#         "offset": label["offset"] + clip_duration,
#         "cluster_id": label["cluster_id"],
#         "cluster": label["cluster"]
#     }
#     audio_clip_list = []
#     label_clip_list = []
#     for pos in range( 0, len(padded_audio), num_samples_in_clip ):
#         ## one clip contains 2 x clip_duration: the first clip_duration is the (left) padded audio part, 
#         ## and the second clip_duration is the main audio part
#         audio_clip = padded_audio[ pos:pos + 2 * num_samples_in_clip]  

#         ## drop too short audios
#         if len(audio_clip) / sr < 0.1:
#             continue
        
#         start_time = pos / sr
#         end_time = (pos + len(audio_clip)) / sr
        
#         intersected_indices = np.logical_and( padded_label["onset"] < end_time, padded_label["offset"] > start_time )
#         label_clip = deepcopy(label)
#         label_clip.update(
#         {
#             "onset": np.maximum(padded_label["onset"][intersected_indices], start_time ) - start_time ,
#             "offset":np.minimum(padded_label["offset"][intersected_indices], end_time ) - start_time ,
#             "cluster_id":padded_label["cluster_id"][intersected_indices],
#             "cluster": [ padded_label["cluster"][idx] for idx in np.argwhere( intersected_indices )[:,0] ]
#         })
        
#         audio_clip_list.append( audio_clip )
#         label_clip_list.append( label_clip )
    
#     assert len(audio_clip_list) == len(label_clip_list)
    
#     return audio_clip_list, label_clip_list


def slice_audio_and_label( audio, label, total_spec_columns ):
    sr = label["sr"]
    clip_duration = total_spec_columns * label["spec_time_step"]
    
    num_samples_in_clip = int( np.round( clip_duration * sr ) )
    padded_audio = np.concatenate( [ np.zeros( ( audio.shape[0], num_samples_in_clip ) ), audio ], axis = 1 ).astype(np.float32)
    
    padded_label = {
        "onset": label["onset"] + clip_duration,
        "offset": label["offset"] + clip_duration,
        "cluster_id": label["cluster_id"],
        "cluster": label["cluster"]
    }
    audio_clip_list = []
    label_clip_list = []
    for pos in range( 0, padded_audio.shape[1], num_samples_in_clip ):
        ## one clip contains 2 x clip_duration: the first clip_duration is the (left) padded audio part, 
        ## and the second clip_duration is the main audio part
        audio_clip = padded_audio[:, pos:pos + 2 * num_samples_in_clip]  

        ## drop too short audios
        if audio_clip.shape[1] / sr < 0.1:
            continue
        
        start_time = pos / sr
        end_time = (pos + audio_clip.shape[1] ) / sr
        
        intersected_indices = np.logical_and( padded_label["onset"] < end_time, padded_label["offset"] > start_time )
        label_clip = deepcopy(label)
        label_clip.update(
        {
            "onset": np.maximum(padded_label["onset"][intersected_indices], start_time ) - start_time ,
            "offset":np.minimum(padded_label["offset"][intersected_indices], end_time ) - start_time ,
            "cluster_id":padded_label["cluster_id"][intersected_indices],
            "cluster": [ padded_label["cluster"][idx] for idx in np.argwhere( intersected_indices )[:,0] ]
        })
        
        audio_clip_list.append( audio_clip )
        label_clip_list.append( label_clip )
    
    assert len(audio_clip_list) == len(label_clip_list)
    
    return audio_clip_list, label_clip_list

# def slice_audios_and_labels( audio_list, label_list, total_spec_columns ):
#     sliced_audio_list, sliced_label_list = [], []
#     for audio, label in zip( audio_list, label_list):
#         sliced_audios, sliced_labels = slice_audio_and_label( audio, label, total_spec_columns )
    
#         sliced_audio_list += sliced_audios
#         sliced_label_list += sliced_labels
    
#     return sliced_audio_list, sliced_label_list

def slice_audios_and_labels( audio_list, label_list, total_spec_columns ):
    sliced_audio_list, sliced_label_list = [], []
    for audio, label in zip( audio_list, label_list):
        sliced_audios, sliced_labels = slice_audio_and_label( audio, label, total_spec_columns )
    
        sliced_audio_list += sliced_audios
        sliced_label_list += sliced_labels
    
    return sliced_audio_list, sliced_label_list

# class VocalSegDataset(Dataset):
#     def __init__(self, audio_list, label_list, tokenizer, max_length, total_spec_columns, species_codebook ):
#         self.audio_list = audio_list
#         self.label_list = label_list
#         self.feature_extractor_bank = self.get_feature_extractor_bank( label_list )
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.total_spec_columns = total_spec_columns
#         self.species_codebook = species_codebook
        
#     def get_feature_extractor_bank(self, label_list ):
#         feature_extractor_bank = {}
#         for label in label_list:
#             key = "%s-%s-%s"%( str( label["sr"] ), str(label["spec_time_step"]), str(label["min_frequency"]) )
#             if key not in feature_extractor_bank:
#                 feature_extractor_bank[key] = WhisperSegFeatureExtractor( label["sr"], label["spec_time_step"], label["min_frequency"] )
#         return feature_extractor_bank
        
#     def map_time_to_spec_col_index(self, t, spec_time_step ):
#         return min( int(np.round( t/( spec_time_step * RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP ) )), self.total_spec_columns  )
        
#     def __len__(self):
#         return len(self.audio_list)
    
#     def __getitem__(self, idx ):
        
#         audio = self.audio_list[idx]
#         label = self.label_list[idx]

#         sr = label["sr"]
#         spec_time_step = label["spec_time_step"]
#         min_frequency = label["min_frequency"]
#         feature_extractor = self.feature_extractor_bank[ "%s-%s-%s"%( str(sr), str(spec_time_step), str(min_frequency) ) ]
        
#         num_samples_in_clip = int(np.round( self.total_spec_columns * spec_time_step * sr ))
                    
#         clip_start = np.random.choice( min( num_samples_in_clip+1, len(audio) - feature_extractor.n_fft + 1 ) )        
#         audio_clip = audio[ clip_start: clip_start + num_samples_in_clip ]
        
#         actual_clip_duration = len( audio_clip ) / sr
#         start_time = clip_start / sr
#         end_time = start_time + actual_clip_duration
        
#         intersected_indices = np.logical_and( label["onset"] < end_time, label["offset"] > start_time )
        
#         onset_in_clip = np.maximum( label["onset"][intersected_indices], start_time ) - start_time 
#         offset_in_clip = np.minimum( label["offset"][intersected_indices], end_time ) - start_time
#         cluster_id_in_clip = label["cluster_id"][intersected_indices]
        
#         """
#         The following code part convert the onset, offset, and cluster_id array into label texts
#         onset_timestamp + cluster_id + offset_timestamp: e.g.,
#         <|zebra_finch|><|0|>7<|6|><|16|>6<|18|>
#         """
#         label_text = [ self.species_codebook.get( label["species"], "<|unknown|>" )  ]
        
#         for pos in range(len(onset_in_clip)):
#             label_text.append( "<|%d|>%d<|%d|>"%(
#                                     self.map_time_to_spec_col_index( onset_in_clip[pos], spec_time_step ),
#                                     cluster_id_in_clip[pos],
#                                     self.map_time_to_spec_col_index( offset_in_clip[pos], spec_time_step ),
#                                 )
#                              )
#         label_text = "".join( label_text )
        
#         audio_clip = np.concatenate( [ audio_clip, np.zeros( num_samples_in_clip - len(audio_clip) ) ], axis = 0 ).astype(np.float32)
#         input_features = feature_extractor(audio_clip, sampling_rate = sr, padding = "do_not_pad")["input_features"][0]
        
#         decoder_input_ids = self.tokenizer.encode( label_text,  max_length = self.max_length + 1, truncation=True, padding = True )
#         labels = decoder_input_ids[1:]
#         decoder_input_ids = decoder_input_ids[:-1]
#         decoder_input_ids += [ self.tokenizer.pad_token_id ] * ( self.max_length - len( decoder_input_ids ) )
#         labels += [-100] * ( self.max_length  - len(labels) )
        
#         return {
#             "input_features":input_features,
#             "decoder_input_ids":np.array(decoder_input_ids),
#             "labels":np.array(labels)
#         }


class VocalSegDataset(Dataset):
    def __init__(self, audio_list, label_list, tokenizer, max_length, total_spec_columns, species_codebook, mask_rate = 0. ):
        self.audio_list = audio_list
        self.label_list = label_list
        self.feature_extractor_bank = self.get_feature_extractor_bank( label_list )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.total_spec_columns = total_spec_columns
        self.species_codebook = species_codebook
        self.mask_rate = mask_rate
        
    def get_feature_extractor_bank(self, label_list ):
        feature_extractor_bank = {}
        for label in label_list:
            key = "%s-%s-%s"%( str( label["sr"] ), str(label["spec_time_step"]), str(label["min_frequency"]) )
            if key not in feature_extractor_bank:
                feature_extractor_bank[key] = WhisperSegFeatureExtractor( label["sr"], label["spec_time_step"], label["min_frequency"] )
        return feature_extractor_bank
        
    def map_time_to_spec_col_index(self, t, spec_time_step ):
        return min( int(np.round( t/( spec_time_step * RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP ) )), self.total_spec_columns  )
        
    def __len__(self):
        return len(self.audio_list)
    
    def __getitem__(self, idx ):
        
        audio = self.audio_list[idx]
        label = self.label_list[idx]

        sr = label["sr"]
        spec_time_step = label["spec_time_step"]
        min_frequency = label["min_frequency"]
        feature_extractor = self.feature_extractor_bank[ "%s-%s-%s"%( str(sr), str(spec_time_step), str(min_frequency) ) ]
        
        num_samples_in_clip = int(np.round( self.total_spec_columns * spec_time_step * sr ))
        clip_start = np.random.choice( min( num_samples_in_clip+1, audio.shape[1] - feature_extractor.n_fft + 1 ) )       
        audio_clip = audio[:, clip_start: clip_start + num_samples_in_clip ].copy()

        ## data augumentation: randomly drop one channels (mask it to zeros)
        # if np.random.rand() < self.mask_rate:
        #     rand_channel_idx = np.random.choice( audio_clip.shape[0] )
        #     audio_clip[rand_channel_idx] = 0.0

        ## scheme 2: randomly introducing partitial mask
        # for channel_idx in range( audio_clip.shape[0] ):
        #     for _ in range(3):
        #         if np.random.rand() < self.mask_rate:
        #             mask_length = int( np.random.rand() * audio_clip.shape[1] * 1.5 )
        #             random_offset = min(
        #                  audio_clip.shape[1],
        #                  max(0, int( np.random.rand() * audio_clip.shape[1] * 1.5 - 0.25 * audio_clip.shape[1] ) )
        #             )
        #             audio_clip[ channel_idx, random_offset:random_offset+mask_length ] = 0.0

        ## scheme 3: randomly drop one channels (mask it to zeros)
        if np.random.rand() < self.mask_rate:
            rand_channel_idx = np.random.choice( audio_clip.shape[0]-1 )  ## do not add mask to mic channel
            if np.random.rand() < 0.5:
                audio_clip[rand_channel_idx] = 0.0
            else:
                mask = np.array( random_mask(audio_clip.shape[1], np.random.rand()) )
                audio_clip[rand_channel_idx] = audio_clip[rand_channel_idx] * mask

        # ## scheme 4: randomly drop one channels (mask it to zeros)
        # if np.random.rand() < self.mask_rate:
        #     rand_channel_idx = 0  ## only add mask to target channel, as the model most make mistakes here
        #     if np.random.rand() < 0.5:
        #         audio_clip[rand_channel_idx] = 0.0
        #     else:
        #         mask = np.array( random_mask(audio_clip.shape[1], np.random.rand()) )
        #         audio_clip[rand_channel_idx] = audio_clip[rand_channel_idx] * mask
                
        actual_clip_duration = audio_clip.shape[1] / sr
        start_time = clip_start / sr
        end_time = start_time + actual_clip_duration
        
        intersected_indices = np.logical_and( label["onset"] < end_time, label["offset"] > start_time )
        
        onset_in_clip = np.maximum( label["onset"][intersected_indices], start_time ) - start_time 
        offset_in_clip = np.minimum( label["offset"][intersected_indices], end_time ) - start_time
        cluster_id_in_clip = label["cluster_id"][intersected_indices]
        
        """
        The following code part convert the onset, offset, and cluster_id array into label texts
        onset_timestamp + cluster_id + offset_timestamp: e.g.,
        <|zebra_finch|><|0|>7<|6|><|16|>6<|18|>
        """
        label_text = [ self.species_codebook.get( label["species"], "<|unknown|>" )  ]
        
        for pos in range(len(onset_in_clip)):
            label_text.append( "<|%d|>%d<|%d|>"%(
                                    self.map_time_to_spec_col_index( onset_in_clip[pos], spec_time_step ),
                                    cluster_id_in_clip[pos],
                                    self.map_time_to_spec_col_index( offset_in_clip[pos], spec_time_step ),
                                )
                             )
        label_text = "".join( label_text )
        
        audio_clip = np.concatenate( [ audio_clip, 
                                       np.zeros( ( audio_clip.shape[0], num_samples_in_clip - audio_clip.shape[1] ) ) ], axis = 1 ).astype(np.float32)
        input_features = []
        for audio_idx in range( audio_clip.shape[0] ):
            input_features.append(feature_extractor(audio_clip[audio_idx], 
                                               sampling_rate = sr, 
                                               padding = "do_not_pad")["input_features"][0]
                                 )
        input_features = np.asarray( input_features )
            
        decoder_input_ids = self.tokenizer.encode( label_text,  max_length = self.max_length + 1, truncation=True, padding = True )
        labels = decoder_input_ids[1:]
        decoder_input_ids = decoder_input_ids[:-1]
        decoder_input_ids += [ self.tokenizer.pad_token_id ] * ( self.max_length - len( decoder_input_ids ) )
        labels += [-100] * ( self.max_length  - len(labels) )
        
        # return {
        #     "input_features":input_features,
        #     "decoder_input_ids":np.array(decoder_input_ids),
        #     "labels":np.array(labels)
        # }

        # ## debugging
        # return {
        #     "input_features":  input_features[0]  ,
        #     "decoder_input_ids":np.array(decoder_input_ids),
        #     "labels":np.array(labels)
        # }

        ## debugging 2: stack three channels but make the overall height still 80
        return {
            "input_features": np.concatenate( [ input_features[c_idx] for c_idx in range( input_features.shape[0] ) ], axis = 0 )[::input_features.shape[0],:] ,
            "decoder_input_ids":np.array(decoder_input_ids),
            "labels":np.array(labels)
        }






