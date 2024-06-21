import os
from glob import glob
import pandas as pd
from transformers import WhisperFeatureExtractor
from transformers.audio_utils import mel_filter_bank
import librosa
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
import json
import re
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.cm as cm


class WhisperSegFeatureExtractor( WhisperFeatureExtractor ):
    def __init__(self, sr, spec_time_step, min_frequency = None, max_frequency = None, chunk_length = 30 ):
        
        hop_length = int( spec_time_step * sr )
        # if hop_length != spec_time_step * sr:
        #     print("Warning: spec_time_step * sr must be an integer. Consider changing the sampling rate sr.")
        
        if sr <= 32000:
            n_fft = 512
        elif sr <= 80000:
            n_fft = 1024
        elif sr <= 150000:
            n_fft = 2048
        elif sr <= 300000:
            n_fft = 4096
        else:
            n_fft = 8192
            
        if min_frequency is None:
            min_frequency = 0
        if max_frequency is None:
            max_frequency = sr // 2
            
        super().__init__(             
            feature_size=80,
            sampling_rate=sr,
            hop_length=hop_length,
            chunk_length = chunk_length,
            n_fft=n_fft,
            padding_value=0.0,
            return_attention_mask=False )
            
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=80,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            sampling_rate=sr,
            norm="slaney",
            mel_scale="slaney",
        )
            
class SpecViewer:
    def __init__( self,  ):
        self.colors = [np.array(mcolors.hex2color(color_string)) for color_string in list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())][1:] # Skip the first color since it looks not so good ...
        unique_colors = None
        for color_arr in self.colors:
            if unique_colors is None:
                unique_colors = np.asarray([color_arr])
            else:
                if np.all( unique_colors == color_arr, axis = 1 ).sum() == 0:
                    unique_colors = np.concatenate( [unique_colors, color_arr[np.newaxis,:]], axis = 0 )
        self.colors = unique_colors[ unique_colors.mean(axis = 1) < 0.8, : ]
        
        self.cmap = cm.get_cmap("magma")
            
    """"
    The following functions are used for implement an interactive visulization function to see the spectrogram and the label
    """

    def chunk_audio(self, audio, start_time, end_time, sr):
        start_idx = int( start_time * sr )
        end_idx = int( end_time * sr )
        chunked_audio = audio[start_idx:end_idx]
        return chunked_audio    

    def chunk_label(self, label, start_time, end_time ):
        
        label_onset_arr = np.array(label["onset"])
        label_offset_arr = np.array(label["offset"])
        
        intersected_indices = np.logical_and( label_onset_arr < end_time, label_offset_arr > start_time )
        chunked_label = {
                "onset": (np.maximum(label_onset_arr[intersected_indices], start_time ) - start_time).tolist(),
                "offset": (np.minimum(label_offset_arr[intersected_indices], end_time ) - start_time).tolist(),
                "cluster": [ label["cluster"][idx] for idx in np.argwhere(intersected_indices)[:,0] ]
            }
        return chunked_label   
    
    def min_max_norm(self, im, min_value = None, max_value = None ):
        if min_value is None:
            min_value = im.min()
        if max_value is None:
            max_value = im.max()
        return (im -  min_value ) / max( max_value - min_value, 1e-12 )

    def plot_spec_and_labels(self, offset, window_size, audio, prediction, label, sr, audio_file_name, feature_extractor, precision_bits , min_spec_value, max_spec_value, xticks_step_size ):
        
        all_unique_clusters = sorted(list(set( list(label["cluster"]) + list(prediction["cluster"]) )))
        cluster_color_mapper = {}
        for cluster in all_unique_clusters:
            if cluster not in cluster_color_mapper:
                cluster_color_mapper[cluster] = self.colors[ len(cluster_color_mapper) % len(self.colors) ]
        
        patches = [Patch(color=color, label=cluster) for cluster, color in cluster_color_mapper.items()]
                
        start_time = offset
        end_time = start_time + window_size
        
        audio_chunked = self.chunk_audio( audio, start_time, end_time, sr )
        label_chunked = self.chunk_label( label, start_time, end_time )
        prediction_chunked = self.chunk_label( prediction, start_time, end_time )
        
        spec = feature_extractor( audio_chunked, sampling_rate=sr, padding = "do_not_pad" )["input_features"][0]
                
        ## convert spec to colorful (3 channel)
        spec_colorful =  np.flip( self.cmap(self.min_max_norm(spec,min_spec_value, max_spec_value))[:,:,:3], axis = 0)
        
        spec_time_step = feature_extractor.hop_length / sr
        spec_xticks_step_size = int(np.round( xticks_step_size / spec_time_step )) 
        spec_xticks_values = np.arange(0, spec.shape[1]+1, spec_xticks_step_size )
        
        # spec_xticks_labels = np.round(spec_xticks_values * spec_time_step + start_time, precision_bits) 
        xticks_format = "%%.%df"%(precision_bits)
        spec_xticks_labels = [ xticks_format%(v) for v in spec_xticks_values * spec_time_step + start_time ]
        
        
        spec_labels_image = np.ones( ( spec.shape[1], 3 ), dtype = np.float32 )
        for pos in range(len(label_chunked["onset"])):
            onset_idx = int(np.round(label_chunked["onset"][pos]/spec_time_step))
            offset_idx = int(np.round(label_chunked["offset"][pos]/spec_time_step)) 
            cluster = label_chunked["cluster"][pos]
            
            ## Add a gap manually if there are two connected segments that have the same cluster but are segmented into two parts (either by human or by machine)
            if pos + 1<len(label_chunked["onset"]) and \
                          offset_idx == int(np.round(label_chunked["onset"][pos+1]/spec_time_step)) and \
                          cluster == label_chunked["cluster"][pos+1]:
                offset_idx -= 1
            
            spec_labels_image[onset_idx:offset_idx,:] = cluster_color_mapper[cluster]
        spec_labels_image = np.tile( spec_labels_image[np.newaxis,:,:], [40,1,1] )
        
        
        spec_preds_image = np.ones( (spec.shape[1], 3), dtype = np.float32 )
        for pos in range(len(prediction_chunked["onset"])):
            onset_idx = int(np.round(prediction_chunked["onset"][pos]/spec_time_step))
            offset_idx = int(np.round(prediction_chunked["offset"][pos]/spec_time_step))
            cluster = prediction_chunked["cluster"][pos]
            
            if pos + 1<len(prediction_chunked["onset"]) and \
                            offset_idx == int(np.round(prediction_chunked["onset"][pos+1]/spec_time_step)) and \
                            cluster == prediction_chunked["cluster"][pos+1]:
                offset_idx -= 1
            
            spec_preds_image[onset_idx:offset_idx,:] = cluster_color_mapper[cluster]
        spec_preds_image = np.tile( spec_preds_image[np.newaxis,:,:], [40,1,1] )
        
        
        canvas_image = np.ones( ( spec_colorful.shape[0] + 10 + 40 + 10 + 40, spec_labels_image.shape[1], 3 ) )
        canvas_image[:spec_colorful.shape[0],:,:] = spec_colorful
        canvas_image[spec_colorful.shape[0]+10:spec_colorful.shape[0]+50,:,:] = spec_preds_image 
        canvas_image[spec_colorful.shape[0]+60:spec_colorful.shape[0]+100,:,:] = spec_labels_image
            
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,4), tight_layout=True, sharex=False)        
        
        ax.imshow( canvas_image, interpolation="bilinear" ) 
        
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.text(-137,35,"Spectrogram:", fontfamily = "monospace" )
        ax.text(-137,-20,"Wav file name: %s"%(audio_file_name), fontfamily = "monospace" )
        ax.text(-137,115,"Prediction:", fontfamily = "monospace" )
        ax.text(-137,165,"Label:", fontfamily = "monospace" )
        ax.set_yticks([])
        ax.set_xticks( spec_xticks_values, spec_xticks_labels )
        ax.set_xlabel("time (s)")
        
        plt.subplots_adjust(wspace=0, hspace=-0.8)    
        plt.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=4)
        plt.show()
        
    def visualize( self, audio, sr, prediction = None, label = None, min_frequency = None, max_frequency = None, precision_bits = 3, audio_file_name = "", window_size = 5.0, xticks_step_size = 0.5, spec_width = 1000):
      
        feature_extractor = WhisperSegFeatureExtractor( sr, window_size / spec_width, min_frequency, max_frequency, chunk_length = max(30, int(np.ceil(window_size)) )  )
        
        
        # whole_spec = feature_extractor( audio, sampling_rate=sr, padding = "do_not_pad" )["input_features"][0]
        min_spec_value = None  # np.percentile( whole_spec, 0.02)
        max_spec_value = None  # np.percentile( whole_spec, 99.98)
        
        if isinstance( label, pd.DataFrame ):
            label_dict = label.to_dict("list")
            
        if isinstance( prediction, pd.DataFrame ):
            prediction = prediction.to_dict("list")
        
        if label is None:
            label = {"onset":[], "offset":[], "cluster":[] }
        if prediction is None:
            prediction = {"onset":[], "offset":[], "cluster":[] }
                
        label["cluster"] = list(map(str, label["cluster"]))
        prediction["cluster"] = list(map(str, prediction["cluster"]))
        
        return interact(self.plot_spec_and_labels, 
                    offset=(0, max(0, len(audio)/sr - window_size ), window_size / 20 ), 
                    window_size = fixed(window_size), 
                    audio = fixed(audio), 
                    prediction = fixed(prediction),
                    label = fixed(label), 
                    sr = fixed(sr), 
                    audio_file_name = fixed(audio_file_name),
                    feature_extractor = fixed(feature_extractor),
                    precision_bits = fixed(precision_bits),
                    min_spec_value = fixed(min_spec_value),
                    max_spec_value = fixed(max_spec_value),
                    xticks_step_size = fixed(xticks_step_size)
                       )

    
def slice_audio_and_label( audio, label, sr, start_time, end_time ):
    sliced_audio = audio[ int( start_time * sr ):int( end_time * sr ) ]
    duration = len(sliced_audio) / sr
    ## get the actual ending time
    end_time = start_time + duration
    
    onsets = np.array( label["onset"] )
    offsets = np.array( label["offset"] )
    clusters = list(label["cluster"])
    
    target_indices = np.argwhere( np.logical_and( onsets < end_time, offsets > start_time ) )[:,0]
    
    sliced_onsets = [ max( 0, onsets[idx] - start_time ) for idx in target_indices ]
    sliced_offsets = [ min( offsets[idx] - start_time, end_time - start_time ) for idx in target_indices ]    
    sliced_clusters = [ clusters[idx] for idx in target_indices ]
    
    sliced_label = {
        "onset":sliced_onsets,
        "offset":sliced_offsets,
        "cluster":sliced_clusters,
    }
    
    if isinstance( label, pd.DataFrame ):
        sliced_label = pd.DataFrame( sliced_label )
    
    return sliced_audio, sliced_label
