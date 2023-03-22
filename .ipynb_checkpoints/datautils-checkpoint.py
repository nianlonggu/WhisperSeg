import os,sys,inspect
import librosa
import pandas as pd
import numpy as np
import threading
from torch.utils.data import Dataset, DataLoader


def load_audio_and_label( audio_path_list, label_path_list,  sr, thread_id, audio_dict, label_dict ):
    local_audio_list = []
    local_label_list = []
    
    for count, (audio_path, label_path) in enumerate(zip( audio_path_list, label_path_list )):
        y, _ = librosa.load( audio_path, sr = sr )
        label_df = pd.read_csv( label_path )
        onset = np.array(label_df["onset"])
        offset = np.array(label_df["offset"])
        local_audio_list.append( y )
        local_label_list.append( {
            "onset":onset,
            "offset":offset
        } )
        
        if count % 10 == 0:
            progress = count / len(audio_path_list)
            print("|%s%s|progress: %.2f %%"%( "-" * (int( progress * 20 )), " "*( 20- int( progress * 20 )), progress*100 ), end = "\r", flush=True)
    
    progress = 1.0
    print("|%s%s|progress: %.2f %%"%( "-" * (int( progress * 20 )), " "*( 20- int( progress * 20 )), progress*100 ), end = "\r", flush=True)
    
    audio_dict[thread_id] = local_audio_list
    label_dict[thread_id] = local_label_list
    
def load_data(audio_path_list, label_path_list, sr, n_threads = 1 ):
    samples_per_thread = int(np.ceil( len(audio_path_list) / n_threads ))
    audio_dict = {}
    label_dict = {}
    thread_list = []
    
    for thread_id, offset in enumerate(range( 0, len(audio_path_list), samples_per_thread )):
        t = threading.Thread( target=load_audio_and_label, args=( audio_path_list[offset:offset+samples_per_thread], 
                                                          label_path_list[offset:offset+samples_per_thread],
                                                          sr,
                                                          thread_id,
                                                          audio_dict, label_dict
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


def split_audio_and_label( audio, label, split_ratio, sr ):
    num_samples_in_audio = len(audio)
    split_point = int( num_samples_in_audio * split_ratio )
    split_time = split_point / sr 
    
    audio_part1 = audio[ :split_point ]
    intersected_indices_part1 = label["onset"] < split_time
    label_part1 = {
        "onset":label["onset"][intersected_indices_part1],
        "offset": np.minimum(label["offset"][intersected_indices_part1], split_time )
    }
    
    audio_part2 = audio[ split_point: ]
    intersected_indices_part2 = label["offset"] > split_time
    label_part2 = {
        "onset": np.maximum(label["onset"][intersected_indices_part2], split_time ) - split_time,
        "offset": label["offset"][intersected_indices_part2] - split_time
    }
    return ( audio_part1, label_part1 ), ( audio_part2, label_part2 )

def train_val_split( audio_list, label_list, val_ratio, sr, n_fft ):
    
    audio_list_train = []
    label_list_train = []
    audio_list_val = []
    label_list_val = []
    
    for audio, label in zip( audio_list, label_list ):
        mode = np.random.choice([0,1])
        if mode == 0:
            (audio_val, label_val), (audio_train, label_train) = split_audio_and_label( audio, label, val_ratio, sr )
        else:
            (audio_train, label_train), (audio_val, label_val) = split_audio_and_label( audio, label, 1-val_ratio, sr )
        
        if len(audio_train) > n_fft:
            audio_list_train.append( audio_train )
            label_list_train.append( label_train )
        
        if len(audio_val) > n_fft:
            audio_list_val.append( audio_val )
            label_list_val.append( label_val )
    
    return (audio_list_train, label_list_train), ( audio_list_val, label_list_val )
        

# def train_val_split( audio_list, label_list, val_ratio ):
#     indices = np.arange(len(audio_list))
#     np.random.shuffle( indices )
#     num_val_examples = int( len(audio_list) * val_ratio )
    
#     val_indices = indices[ :num_val_examples ]
#     train_indices = indices[ num_val_examples: ]
    
#     audio_list_train = [audio_list[idx] for idx in train_indices]
#     label_list_train = [label_list[idx] for idx in train_indices]
    
#     audio_list_val = [audio_list[idx] for idx in val_indices]
#     label_list_val = [label_list[idx] for idx in val_indices]
    
#     return (audio_list_train, label_list_train), ( audio_list_val, label_list_val )


def slice_audio_and_label( audio, label, clip_duration, sr, n_fft ):
    num_samples_in_clip = int( np.round( clip_duration * sr ) )
    padded_audio = np.concatenate( [ np.zeros( num_samples_in_clip ), audio ], axis = 0 )
    padded_label = {
        "onset": label["onset"] + clip_duration,
        "offset": label["offset"] + clip_duration
    }
    
    audio_clip_list = []
    label_clip_list = []
    for pos in range( 0, len(padded_audio), num_samples_in_clip ):
        ## one clip contains 2 x clip_duration: the first clip_duration is the (left) padded audio part, 
        ## and the second clip_duration is the main audio part
        audio_clip = padded_audio[ pos:pos + 2 * num_samples_in_clip]  
        
        if len(audio_clip) <= n_fft:
            continue
        
        start_time = pos / sr
        end_time = (pos + len(audio_clip)) / sr
        
        intersected_indices = np.logical_and( padded_label["onset"] < end_time, padded_label["offset"] > start_time )
        label_clip = {
            "onset": np.maximum(padded_label["onset"][intersected_indices], start_time ) - start_time ,
            "offset":np.minimum(padded_label["offset"][intersected_indices], end_time ) - start_time
        }
        
        audio_clip_list.append( audio_clip )
        label_clip_list.append( label_clip )
    
    assert len(audio_clip_list) == len(label_clip_list)
    
    return audio_clip_list, label_clip_list

def slice_audios_and_labels( audio_list, label_list, clip_duration, sr, n_fft ):
    sliced_audio_list, sliced_label_list = [], []
    for audio, label in zip( audio_list, label_list):
        sliced_audios, sliced_labels = slice_audio_and_label( audio, label, clip_duration, sr, n_fft )
    
        sliced_audio_list += sliced_audios
        sliced_label_list += sliced_labels
    
    return sliced_audio_list, sliced_label_list

class VocalSegDataset(Dataset):
    def __init__(self, audio_list, label_list, feature_extractor, tokenizer, max_length, 
                       sr, clip_duration, input_features_length, timestamp_precision, timestamp_format ):
        self.audio_list = audio_list
        self.label_list = label_list
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sr = sr
        self.clip_duration = clip_duration
        self.input_features_length = input_features_length
        self.timestamp_precision = timestamp_precision
        self.timestamp_format = timestamp_format
        
        self.num_padding_hops = int( np.ceil( feature_extractor.n_fft / feature_extractor.hop_length ) )
        self.n_padding = self.num_padding_hops * feature_extractor.hop_length
        
    def quantize(self, arr):
        return np.round(arr / self.timestamp_precision) * self.timestamp_precision 
        
    def __len__(self):
        return len(self.audio_list)
    
    def __getitem__(self, idx ):
        
        audio = self.audio_list[idx]
        label = self.label_list[idx]
        onset = label["onset"]
        offset = label["offset"]
        
        num_samples_in_clip = int(np.round( self.clip_duration * self.sr ))
        clip_start = np.random.choice( min( num_samples_in_clip+1, len(audio) - self.feature_extractor.n_fft + 1 ) )        
        audio_clip = audio[ clip_start: clip_start + num_samples_in_clip ]
        
        actual_clip_duration = len( audio_clip ) / self.sr
        start_time = clip_start / self.sr
        end_time = start_time + actual_clip_duration
        
        intersected_indices = np.logical_and( label["onset"] < end_time, label["offset"] > start_time )
        onset_in_clip = np.maximum( label["onset"][intersected_indices], start_time ) - start_time 
        offset_in_clip = np.minimum( label["offset"][intersected_indices], end_time ) - start_time
        
        """
        The following code part convert the onset and offset array into label texts
        """
        label_text = []
        for pos in range(len(onset_in_clip)):
            text = []
            if onset_in_clip[pos] > 0 :
                text.append( self.timestamp_format%( self.quantize(onset_in_clip[pos]) ))
            text.append("IN")
            if offset_in_clip[pos] < actual_clip_duration:
                text.append( self.timestamp_format%( self.quantize(offset_in_clip[pos]) ) )
            label_text.append( "".join(text) )
            
        if len(onset_in_clip) >0:
            if onset_in_clip[0] > 0 :
                label_text = [""] + label_text
            if offset_in_clip[-1] < actual_clip_duration:
                label_text += [""]
        else:
            label_text = ["",""]
        label_text = "OUT".join(label_text)
        """********"""
        
        input_features = self.feature_extractor(audio_clip, sampling_rate = self.sr, padding = "do_not_pad")["input_features"][0]
        feature_padding_value = input_features.min() if input_features.shape[1] != 0 else -1.0
        input_features = input_features[:,:self.input_features_length]
        input_features = np.concatenate([ input_features, 
                                          feature_padding_value *np.ones( ( input_features.shape[0], self.input_features_length - input_features.shape[1]) )
                                        ],
                                          axis = 1
                                       ).astype(np.float32)
        
        decoder_input_ids = self.tokenizer.encode( label_text,  max_length = self.max_length + 1, truncation=True, padding = True )
        labels = decoder_input_ids[1:]
        decoder_input_ids = decoder_input_ids[:-1]
        decoder_input_ids += [ self.tokenizer.pad_token_id ] * ( self.max_length - len( decoder_input_ids ) )
        labels += [-100] * ( self.max_length  - len(labels) )
        
        return {
            "input_features":input_features,
            "decoder_input_ids":np.array(decoder_input_ids),
            "labels":np.array(labels),
            # "onset_in_clip":onset_in_clip,
            # "offset_in_clip":offset_in_clip
        }