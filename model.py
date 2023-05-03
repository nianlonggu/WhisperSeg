import os
import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration
from glob import glob
import re
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
import ctranslate2
import json
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.cm as cm


from PIL import Image

def load_model( model_folder, initial_model_path, sr, hop_length, input_features_length, 
                timestamp_format, timestamp_precision, clip_duration, dropout = 0.0):
    ckpt_list =  glob( model_folder + "/*" )
    if len( ckpt_list ) >0:
        ckpt_list.sort( key = os.path.getmtime )
        ckpt_name = ckpt_list[-1]
        current_batch = int(ckpt_name.split("-")[-1])
        model = WhisperForConditionalGeneration.from_pretrained(ckpt_name)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(ckpt_name)
        tokenizer = WhisperTokenizer.from_pretrained(ckpt_name, language = "english" )
        print("Model loaded!")
    else:
        current_batch = 0
        model = WhisperForConditionalGeneration.from_pretrained(initial_model_path)
        ## chunk the position embedding to 0.5 * input_features_length. This can help to save memory usage and speed up training!
        model.config.max_source_positions = int( 0.5*input_features_length )
        with torch.no_grad():
            model.model.encoder.embed_positions.weight = torch.nn.Parameter(
                          model.model.encoder.embed_positions.weight[:model.config.max_source_positions,:]
            )
            model.model.encoder.embed_positions.num_embeddings = model.config.max_source_positions
        
        model.config.clip_duration = clip_duration
        model.config.sr = sr
        model.config.timestamp_format = timestamp_format
        model.config.timestamp_precision = timestamp_precision
                    
        model.config.dropout = dropout
        model.model.encoder.dropout = dropout
        for layer in model.model.encoder.layers:
            layer.dropout = dropout
        model.model.decoder.dropout = dropout
        for layer in model.model.decoder.layers:
            layer.dropout = dropout
            
        model.config.cluster_codebook = {}

        feature_extractor = WhisperFeatureExtractor(  
            feature_size=80,
            sampling_rate=sr,
            hop_length=hop_length,
            chunk_length=30,
            n_fft=400,
            padding_value=0.0,
            return_attention_mask=False)
        
        ## tokeners for openai/whisper-large openai/whisper-base openai/whisper-small ... are all the same!
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", language = "english" )
        # tokenizer.add_special_tokens( { 'additional_special_tokens': [timestamp_format%(i*timestamp_precision) for i in range(1500+1)] } )
        ## tokenizer.add_special_tokens will change all_special_ids, while tokenizer.add_tokens([XXX,], special_tokens=True ) does not. 
        ## change of all_special_ids will result in the change of encoding behavior
        tokenizer.add_tokens( [timestamp_format%(i*timestamp_precision) for i in range(1500+1)], special_tokens=True  )
        
    return model, feature_extractor, tokenizer, current_batch


def save_model( model, feature_extractor, tokenizer, current_batch, model_folder, max_to_keep ):
    try:
        model = model.module
    except:
        pass
    
    ckpt_list =  glob( model_folder + "/*" )
    feature_extractor.save_pretrained( model_folder+"/checkpoint-%d"%(current_batch) )
    tokenizer.save_pretrained( model_folder+"/checkpoint-%d"%(current_batch) )
    model.save_pretrained( model_folder+"/checkpoint-%d"%(current_batch) )
    
    if max_to_keep > 0 and len( ckpt_list ) > max_to_keep:
        ckpt_list.sort( key = os.path.getmtime )
        ckpt_name = ckpt_list[0]
        os.system("rm -r %s"%(ckpt_name) )
        

        

class SegmenterBase:
    def __init__( self,  ):
        self.segment_matcher = re.compile("<\|([0-9]+\.[0-9]+)\|>(\d+?)<\|([0-9]+\.[0-9]+)\|>")
        self.colors = [np.array(mcolors.hex2color(color_string)) for color_string in list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())][1:] # Skip the first color since it looks not so good ...
        
        unique_colors = None
        for color_arr in self.colors:
            if unique_colors is None:
                unique_colors = np.asarray([color_arr])
            else:
                if np.all( unique_colors == color_arr, axis = 1 ).sum() == 0:
                    unique_colors = np.concatenate( [unique_colors, color_arr[np.newaxis,:]], axis = 0 )
        self.colors = unique_colors[ unique_colors.mean(axis = 1) < 0.8, : ]
        
        
        self.cluster_codebook = None
        self.inverse_cluster_codebook = None
        self.cmap = cm.get_cmap("jet")
        
        
        ### These paremeters are used for voting the predictions of multiple trials to get the final prediction
        self.noise_cluster = ">>>__NOISE_SEG_CLUSTER__<<<"
        self.cluster_counter_sufix = ">>>__CLUSTER_COUNTER_%d__<<<"
        self.cluster_counter_sufix_matcher = re.compile(">>>__CLUSTER_COUNTER_\d+__<<<")
    
    ### segmentation-related functions:
    def extract_segments( self, text ):
        segment_list = []
        match_res_list = self.segment_matcher.findall( text )
        for onset_text, cluster_id_text, offset_text in match_res_list:
            try:
                onset = float( onset_text )
                offset = float( offset_text )
                cluster_id = int( cluster_id_text )
                assert cluster_id in self.inverse_cluster_codebook
                assert offset - onset > 0
                cluster = self.inverse_cluster_codebook[cluster_id]
                segment_list.append( [ onset, offset, cluster ] )
            except:
                continue
        return segment_list
    
    def select_prediction_given_time_range( self, prediction, start_time, end_time):
        
        selected_prediction = []
        for onset, offset, cluster in prediction:
            if onset < end_time and offset > start_time:
                selected_prediction.append( [ onset, offset, cluster ] )
        if len(selected_prediction) > 0 and selected_prediction[-1][1] > end_time:
            is_cutoff = True
        else:
            is_cutoff = False
        
        cluster_counter = {}
        
        selected_prediction_with_noise_seg = []
        current_time = start_time
        for onset, offset, cluster in selected_prediction:
            onset = max( onset, current_time )
            offset = min( offset, end_time )
            if onset > current_time:
                selected_prediction_with_noise_seg.append([ current_time, onset, self.noise_cluster ])
            if onset < offset:
                
                #cluster_counter[cluster] = cluster_counter.get(cluster, 0) + 1
                """
                This is the trick that is used to create a separate sub-name for two same adjacent clusters. 
                For example, the processed cluster name will look like "seg_cluster_1_count_1 | seg_cluster_1_count_2". 
                This set up was originalally used for the case where two adjacent segments of the same type are very close (and even 0 gap),
                but it sometimes bring unexpected behavior. Disable is for now (by setting the subname always to 0).
                """
                cluster_counter[cluster] = 0
                selected_prediction_with_noise_seg.append([ onset, offset, cluster + self.cluster_counter_sufix%(cluster_counter[cluster]) ])
                current_time = offset
            
            if offset >= end_time:
                break
        if current_time < end_time:
            selected_prediction_with_noise_seg.append([ current_time, end_time, self.noise_cluster ])
        
        return selected_prediction_with_noise_seg, is_cutoff

    
    def vote_predictions(self, selected_prediction_list, voting_precision ):
        assert len(selected_prediction_list) > 0
        start_time = selected_prediction_list[0][0][0]
        end_time = selected_prediction_list[0][-1][1]
        
        assert end_time - start_time > voting_precision
        
        for sel_pred in selected_prediction_list:
            assert len(sel_pred) > 0
            assert sel_pred[0][0] == start_time and sel_pred[-1][1] == end_time
        
        cluster_row_id_mapper = {}
        for sel_pred in selected_prediction_list:
            for onset, offset, cluster in sel_pred:
                if cluster not in cluster_row_id_mapper:
                    cluster_row_id_mapper[cluster] = len(cluster_row_id_mapper)
        row_id_cluster_mapper = { row_id:cluster for cluster, row_id in cluster_row_id_mapper.items() }
        
        prediction_matrix = np.zeros( (len(cluster_row_id_mapper), int(np.round( (end_time - start_time)/voting_precision )) ))
        for sel_pred in selected_prediction_list:
            for onset, offset, cluster in sel_pred:
                row_id = cluster_row_id_mapper[cluster]
                onset_pos = int(np.round( (onset - start_time)/voting_precision))
                offset_pos = int(np.round( (offset - start_time)/voting_precision ))
                
                ## Trick: Give the noise cluster slightly higher weight, so that at a certain timestamp, 
                ## if all trials produce different clusters and one trial predict noise at that point, then the voted label will be noise
                prediction_matrix[ row_id, onset_pos:offset_pos ] += (1.5 if  cluster == self.noise_cluster else 1)
                
        prediction_voted = np.argmax( prediction_matrix, axis = 0 ) 
    
        voted_segments = []    
        boundaries = np.argwhere( np.array(prediction_voted.tolist()+[-1]) - np.array([-1]+prediction_voted.tolist()) )[:,0]
        for pos in range( len(boundaries)-1 ):
            onset = boundaries[pos]
            offset = boundaries[pos+1]
            row_id = prediction_voted[onset]
            onset_time = onset * voting_precision + start_time
            offset_time = offset * voting_precision + start_time
            cluster = row_id_cluster_mapper[row_id]
            
            if cluster == self.noise_cluster:
                continue
            
            cluster = self.cluster_counter_sufix_matcher.sub("", cluster)
            voted_segments.append([ onset_time, min(offset_time, end_time), cluster ])
            
        return voted_segments
    
    
    def compute_syllable_score( self, prediction_on_offset_list, label_on_offset_list, tolerance = 0.02  ):
        
        n_positive_in_prediction = len(prediction_on_offset_list)
        n_positive_in_label = len(label_on_offset_list)
        
        n_true_positive = 0
        for pred_onset, pred_offset in prediction_on_offset_list:
            is_matched = False
            for count, (label_onset, label_offset) in enumerate( label_on_offset_list ):
                if np.abs( pred_onset -  label_onset )<=tolerance and np.abs( pred_offset - label_offset )<= tolerance:
                    # print( (pred_onset, pred_offset), (label_onset, label_offset) )
                    n_true_positive += 1
                    is_matched = True
                    break  # early stop for the predicted value
            if is_matched:
                ## remove the already matched syllable from the ground-truth
                label_on_offset_list.pop(count)
        
        return n_true_positive, n_positive_in_prediction, n_positive_in_label
        
    
    def score( self, prediction, label, target_cluster = None, tolerance = 0.02 ):
        
        prediction_on_offset_list = []
        for pos in range(len(prediction["onset"])):
            if target_cluster is None or str(target_cluster) == str(prediction["cluster"][pos]):
                prediction_on_offset_list.append([ prediction["onset"][pos], prediction["offset"][pos] ])
        
        label_on_offset_list = []
        for pos in range(len(label["onset"])):
            if target_cluster is None or str(target_cluster) == str( label["cluster"][pos] ):
                label_on_offset_list.append([ label["onset"][pos], label["offset"][pos] ])

        if target_cluster is not None and len(label_on_offset_list) == 0:
            print("Warning: the specified target cluster '%s' does not exist in the ground-truth labels."%(str(target_cluster)))
        
        TP, P_pred, P_label = self.compute_syllable_score( prediction_on_offset_list, label_on_offset_list, tolerance  )
            
        precision = TP / max(P_pred, 1e-12  )
        recall = TP / max( P_label, 1e-12 )
        f1 = 2/(1/ max(precision, 1e-12) + 1/max(recall, 1e-12)  )
            
        return TP, P_pred, P_label, precision, recall, f1
        
    
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
    
    def min_max_norm(self, im):
        return (im - im.min()) / max( im.max() - im.min(), 1e-12 )
        

    def plot_spec_and_labels(self, offset, window_size, audio, prediction, label, sr, audio_file_name, feature_extractor ):
        
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
        
        spec = feature_extractor( audio_chunked, sampling_rate=sr, padding = "do_not_pad")["input_features"][0]
        
        ## convert spec to colorful (3 channel)
        spec_colorful =  np.flip( self.cmap(self.min_max_norm(spec))[:,:,:3], axis = 0)
        
        time_per_column = feature_extractor.hop_length / sr
        spec_xticks_step_size = int(np.round( 0.5 / time_per_column )) 
        spec_xticks_values = np.arange(0, spec.shape[1]+1, spec_xticks_step_size )
        
        precision_bits = int(re.findall( "\%\.(\d)f", self.timestamp_format)[0])
        # spec_xticks_labels = np.round(spec_xticks_values * time_per_column + start_time, precision_bits) 
        xticks_format = "%%.%df"%(precision_bits)
        spec_xticks_labels = [ xticks_format%(v) for v in spec_xticks_values * time_per_column + start_time ]
        
        
        spec_labels_image = np.ones( ( spec.shape[1], 3 ), dtype = np.float32 )
        for pos in range(len(label_chunked["onset"])):
            onset_idx = int(np.round(label_chunked["onset"][pos]/time_per_column))
            offset_idx = int(np.round(label_chunked["offset"][pos]/time_per_column)) 
            cluster = label_chunked["cluster"][pos]
            
            ## Add a gap manually if there are two connected segments that have the same cluster but are segmented into two parts (either by human or by machine)
            if pos + 1<len(label_chunked["onset"]) and \
                          offset_idx == int(np.round(label_chunked["onset"][pos+1]/time_per_column)) and \
                          cluster == label_chunked["cluster"][pos+1]:
                offset_idx -= 1
            
            spec_labels_image[onset_idx:offset_idx,:] = cluster_color_mapper[cluster]
        spec_labels_image = np.tile( spec_labels_image[np.newaxis,:,:], [40,1,1] )
        
        
        spec_preds_image = np.ones( (spec.shape[1], 3), dtype = np.float32 )
        for pos in range(len(prediction_chunked["onset"])):
            onset_idx = int(np.round(prediction_chunked["onset"][pos]/time_per_column))
            offset_idx = int(np.round(prediction_chunked["offset"][pos]/time_per_column))
            cluster = prediction_chunked["cluster"][pos]
            
            if pos + 1<len(prediction_chunked["onset"]) and \
                            offset_idx == int(np.round(prediction_chunked["onset"][pos+1]/time_per_column)) and \
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
        
    def visualize( self, audio, prediction = None, label = None, audio_file_name = "", window_size = 5.0, image_width = 1000):
        sr = self.sr
        if window_size is None:
            window_size = self.clip_duration

        time_per_col = window_size / image_width
        hop_length = int( time_per_col * sr )
        feature_extractor = WhisperFeatureExtractor(  
            feature_size=80,
            sampling_rate=sr,
            hop_length=hop_length,
            chunk_length=30,
            n_fft=400,
            padding_value=0.0,
            return_attention_mask=False)
        
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
                    feature_extractor = fixed(feature_extractor)
                       )
    

    
class WhisperSegmenter(SegmenterBase):        
    def __init__(self, model_path = None, device = None, model = None, feature_extractor = None, tokenizer = None):
        super().__init__()
        if model_path is not None:
            self.model = WhisperForConditionalGeneration.from_pretrained( model_path )
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained( model_path )
            self.tokenizer = WhisperTokenizer.from_pretrained(model_path, language = "english" )
            if device is not None:
                self.model = self.model.to(device)
        else:
            try:
                self.model = model.module
            except:
                self.model = model
            
            self.feature_extractor = feature_extractor
            self.tokenizer = tokenizer
        
        self.device = list( self.model.parameters() )[0].device
        
        # here 2 is determined by the conv kernel of the whisper
        self.sr = self.model.config.sr
        self.clip_duration = self.model.config.clip_duration
        self.timestamp_precision = self.model.config.timestamp_precision
        self.timestamp_format = self.model.config.timestamp_format
        self.precision_bits = int(re.findall( "\%\.(\d)f", self.timestamp_format)[0])
        self.input_features_length = int(self.clip_duration/ (self.feature_extractor.hop_length / self.sr))

        self.cluster_codebook = self.model.config.cluster_codebook
        self.inverse_cluster_codebook = { cluster_id:cluster  for cluster, cluster_id in self.cluster_codebook.items() }
            
    
    
    def update_cluster_codebook(self, cluster_codebook):
        self.model.config.cluster_codebook = cluster_codebook
        
        self.cluster_codebook = cluster_codebook
        self.inverse_cluster_codebook = { cluster_id:cluster for cluster, cluster_id in self.cluster_codebook.items() }
                
            
        
    @torch.no_grad()
    def segment( self, audio, num_trials = 3, min_segment_length = 0.02,
                       voting_time_step = 1.0, voting_precision = 0.001, 
                       batch_size = 16, max_length = 448
               ):
        ## voting_time_step is used to during voting for the final prediction based on multiple trials. 
        ## set it to a smaller value if the segments are very dense with small gaps
        
        clip_duration = self.clip_duration
        sr = self.sr
        
        voting_precision = min( voting_precision, self.timestamp_precision )
        
        max_num_padding_samples = int( clip_duration * sr )
        audio_left_pad = np.zeros( max_num_padding_samples, dtype = np.float32 )
        audio_clip_length = int(clip_duration * sr)
        
        sliced_audios_features = []
        for trial_id in range(num_trials):            
            
            padding_time = np.round( clip_duration * trial_id / num_trials / self.timestamp_precision ) * self.timestamp_precision
            num_padding_samples = int( padding_time * sr )
                        
            audio_padded = np.concatenate( [ audio_left_pad[ len(audio_left_pad) - num_padding_samples: ],
                                             audio
                                           ], axis = 0 )
            
            ## This loop must be executed once even for zero length audio
            for pos in range( 0, max(len(audio_padded), 1), audio_clip_length ):
                offset_time = pos / sr - padding_time
                
                audio_clip = audio_padded[pos:pos+audio_clip_length]
                audio_clip_padded = np.concatenate([ audio_clip, np.zeros( audio_clip_length - len(audio_clip), dtype = np.float32 ) ], axis = 0 )
            
                input_features = self.feature_extractor(audio_clip_padded, sampling_rate = sr, padding = "do_not_pad")["input_features"][0]
                feature_padding_value = input_features.min() if input_features.shape[1] != 0 else -1.0
                input_features = input_features[:,:self.input_features_length] 
                input_features = np.concatenate([ 
                                    input_features, 
                                    feature_padding_value * np.ones( ( input_features.shape[0], self.input_features_length - input_features.shape[1]) )
                                  ], axis = 1 ).astype(np.float32)
                
                sliced_audios_features.append( ( trial_id, offset_time, input_features, len(audio_clip)/sr ) )
                
        ### the generation part
        generated_text_list = []
        for pos in range( 0, len(sliced_audios_features), batch_size ):
            
            # This is the code if self.model is the raw huggingface model
            input_features = torch.from_numpy( np.asarray([ item[2] for item in sliced_audios_features[pos:pos+batch_size] ]) ).to(self.device)
            generated_ids = self.model.generate( inputs = input_features, max_length = max_length)
            generated_text_batch = self.tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
            
            generated_text_list += generated_text_batch
            
        ## convert generated text to on_offsets
        on_offset_list_of_trial = {}
        for count, generated_text in enumerate(generated_text_list):
            trial_id, offset_time, _, duration = sliced_audios_features[count]

            if trial_id not in on_offset_list_of_trial:
                on_offset_list_of_trial[trial_id] = []            

            on_offsets_clip = self.extract_segments( generated_text )
            for item in on_offsets_clip:
                item[0] += offset_time
                item[1] += offset_time

            on_offset_list_of_trial[trial_id].append( on_offsets_clip )
        
        ## merge (or concatenate) the on_offset of each trial separately
        merged_on_offset_list_of_trial = {}
        for trial_id in on_offset_list_of_trial:
            merged_on_offset_list_of_trial[trial_id] = []
            for on_offsets_clip in on_offset_list_of_trial[trial_id]:
                
                if len(merged_on_offset_list_of_trial[trial_id]) > 0 and \
                   len(on_offsets_clip)>0 and \
                   merged_on_offset_list_of_trial[trial_id][-1][1] == on_offsets_clip[0][0] and \
                   merged_on_offset_list_of_trial[trial_id][-1][2] == on_offsets_clip[0][2]:
                    
                    merged_on_offset_list_of_trial[trial_id][-1][1] = on_offsets_clip[0][1]
                    on_offsets_clip = on_offsets_clip[1:]
                merged_on_offset_list_of_trial[trial_id] += on_offsets_clip
        
        for trial_id in merged_on_offset_list_of_trial:
            merged_on_offset_list_of_trial[trial_id] = sorted( merged_on_offset_list_of_trial[trial_id], key = lambda x:x[0] )
            merged_on_offset_list_of_trial[trial_id] = [ item for item in merged_on_offset_list_of_trial[trial_id] if item[1] - item[0] > 0 ]
        
        ## obtain the final prediction by voting
        ## NOTE: This voting is done piece by piece
        
        voted_segments = []
        prev_is_cutoff = False
        total_audio_duration = len(audio) / sr
        
        for start_time in np.arange( 0, total_audio_duration, voting_time_step ):
            end_time = start_time + voting_time_step
            
            selected_prediction_list, is_cutoff_list = zip(*[ self.select_prediction_given_time_range(merged_on_offset_list_of_trial[trial_id], start_time, end_time ) for trial_id in merged_on_offset_list_of_trial ])
            is_cutoff = np.mean(is_cutoff_list) > 0.5
            
            voted_selected_segments = self.vote_predictions( selected_prediction_list, voting_precision )
            
            ## try two joint two segments that belong to one
            if  prev_is_cutoff and \
                len(voted_segments) > 0 and \
                len(voted_selected_segments) > 0 and \
                voted_segments[-1][2] == voted_selected_segments[0][2] and \
                np.abs( voted_segments[-1][1] - voted_selected_segments[0][0] ) < 1e-6:
                voted_segments[-1][1] = voted_selected_segments[0][1]
                voted_selected_segments = voted_selected_segments[1:]
            
            voted_segments += voted_selected_segments
            prev_is_cutoff = is_cutoff
            
        if min_segment_length is not None:
            voted_segments = [ item for item in voted_segments if item[1] - item[0] >= min_segment_length  ]

            
        if len(voted_segments) == 0:
            pred_onsets, pred_offsets, pred_clusters = [], [], []
        else:
            pred_onsets, pred_offsets, pred_clusters = list(zip(*voted_segments  ))
        
        return {"onset":list(pred_onsets), "offset":list(pred_offsets), "cluster":list(pred_clusters) }


"""
WhisperSegmenterFast differs from WhisperSegmenter in
1. the init function
2. the text generation part in the segment function
3. WhisperSegmenterFast does not need the function update_cluster_codebook, because it is not used for training with new data.
"""    
    
class WhisperSegmenterFast(SegmenterBase):
    def __init__(self, model_path, device="cpu", compute_type = "float16" ):
        super().__init__()
        
        
        self.model = ctranslate2.models.Whisper(model_path, device = device, compute_type = compute_type)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained( model_path+"/hf_model" )
        self.tokenizer = WhisperTokenizer.from_pretrained(model_path+"/hf_model", language = "english" )
        
        model_config = json.load(open(model_path+"/hf_model"+"/config.json"))
         

        self.sr = model_config["sr"]
        self.clip_duration = model_config["clip_duration"]
        self.timestamp_precision = model_config["timestamp_precision"]
        self.timestamp_format = model_config["timestamp_format"]
        self.precision_bits = int(re.findall( "\%\.(\d)f", self.timestamp_format)[0])
        self.input_features_length = int(self.clip_duration/ (self.feature_extractor.hop_length / self.sr))

        self.cluster_codebook = model_config["cluster_codebook"]
        self.inverse_cluster_codebook = { cluster_id:cluster  for cluster, cluster_id in self.cluster_codebook.items() }
            
        
    @torch.no_grad()
    def segment( self, audio, num_trials = 3, min_segment_length = 0.02,
                       voting_time_step = 1.0, voting_precision = 0.001, 
                       batch_size = 16, max_length = 448
               ):
        

        clip_duration = self.clip_duration
        sr = self.sr
        
        voting_precision = min( voting_precision, self.timestamp_precision )
        
        max_num_padding_samples = int( clip_duration * sr )
        audio_left_pad = np.zeros( max_num_padding_samples, dtype = np.float32 )
        audio_clip_length = int(clip_duration * sr)
        
        sliced_audios_features = []
        for trial_id in range(num_trials):            
            
            padding_time = np.round( clip_duration * trial_id / num_trials / self.timestamp_precision ) * self.timestamp_precision
            num_padding_samples = int( padding_time * sr )
                        
            audio_padded = np.concatenate( [ audio_left_pad[ len(audio_left_pad) - num_padding_samples: ],
                                             audio
                                           ], axis = 0 )
            
            ## This loop must be executed once even for zero length audio
            for pos in range( 0, max(len(audio_padded), 1), audio_clip_length ):
                offset_time = pos / sr - padding_time
                
                audio_clip = audio_padded[pos:pos+audio_clip_length]
                audio_clip_padded = np.concatenate([ audio_clip, np.zeros( audio_clip_length - len(audio_clip), dtype = np.float32 ) ], axis = 0 )
            
                input_features = self.feature_extractor(audio_clip_padded, sampling_rate = sr, padding = "do_not_pad")["input_features"][0]
                feature_padding_value = input_features.min() if input_features.shape[1] != 0 else -1.0
                input_features = input_features[:,:self.input_features_length] 
                input_features = np.concatenate([ 
                                    input_features, 
                                    feature_padding_value * np.ones( ( input_features.shape[0], self.input_features_length - input_features.shape[1]) )
                                  ], axis = 1 ).astype(np.float32)
                
                sliced_audios_features.append( ( trial_id, offset_time, input_features, len(audio_clip)/sr ) )
                        
                
        ### the generation part
        generated_text_list = []
        for pos in range( 0, len(sliced_audios_features), batch_size ):
            
            """ 
            This is the code if self.model is the converted ctranslate model
            
            """
            sliced_audios_features_batch = sliced_audios_features[pos:pos+batch_size]
            actual_batch_size = len(sliced_audios_features_batch)
            features = ctranslate2.StorageView.from_array(np.asarray([ item[2] for item in sliced_audios_features_batch ]))
            prompt = self.tokenizer.convert_tokens_to_ids(
                [ "<|startoftranscript|>", "<|en|>", "<|notimestamps|>"]
            )
            ## the ctranslate converted model typically requires a larger max length than the one required by the original huggingface model, so we set max_length to a large value.
            model_output = self.model.generate(features, [ prompt for _ in range(actual_batch_size) ], max_length = max_length )
            generated_text_batch = []
            for item in model_output:
                try:
                    gen_text = "".join(item.sequences[0])
                except:
                    gen_text = ""
                generated_text_batch.append(gen_text)
                
            generated_text_list += generated_text_batch
            
        ## convert generated text to on_offsets
        on_offset_list_of_trial = {}
        for count, generated_text in enumerate(generated_text_list):
            trial_id, offset_time, _, duration = sliced_audios_features[count]

            if trial_id not in on_offset_list_of_trial:
                on_offset_list_of_trial[trial_id] = []            

            on_offsets_clip = self.extract_segments( generated_text )
            for item in on_offsets_clip:
                item[0] += offset_time
                item[1] += offset_time

            on_offset_list_of_trial[trial_id].append( on_offsets_clip )
        
        ## merge (or concatenate) the on_offset of each trial separately
        merged_on_offset_list_of_trial = {}
        for trial_id in on_offset_list_of_trial:
            merged_on_offset_list_of_trial[trial_id] = []
            for on_offsets_clip in on_offset_list_of_trial[trial_id]:
                
                if len(merged_on_offset_list_of_trial[trial_id]) > 0 and \
                   len(on_offsets_clip)>0 and \
                   merged_on_offset_list_of_trial[trial_id][-1][1] == on_offsets_clip[0][0] and \
                   merged_on_offset_list_of_trial[trial_id][-1][2] == on_offsets_clip[0][2]:
                    
                    merged_on_offset_list_of_trial[trial_id][-1][1] = on_offsets_clip[0][1]
                    on_offsets_clip = on_offsets_clip[1:]
                merged_on_offset_list_of_trial[trial_id] += on_offsets_clip
        
        for trial_id in merged_on_offset_list_of_trial:
            merged_on_offset_list_of_trial[trial_id] = sorted( merged_on_offset_list_of_trial[trial_id], key = lambda x:x[0] )
            merged_on_offset_list_of_trial[trial_id] = [ item for item in merged_on_offset_list_of_trial[trial_id] if item[1] - item[0] > 0 ]
        
        ## obtain the final prediction by voting
        ## NOTE: This voting is done piece by piece
        
        voted_segments = []
        prev_is_cutoff = False
        total_audio_duration = len(audio) / sr
        
        for start_time in np.arange( 0, total_audio_duration, voting_time_step ):
            end_time = start_time + voting_time_step
            
            selected_prediction_list, is_cutoff_list = zip(*[ self.select_prediction_given_time_range(merged_on_offset_list_of_trial[trial_id], start_time, end_time ) for trial_id in merged_on_offset_list_of_trial ])
            is_cutoff = np.mean(is_cutoff_list) > 0.5
            
            voted_selected_segments = self.vote_predictions( selected_prediction_list, voting_precision )
            
            ## try two joint two segments that belong to one
            if  prev_is_cutoff and \
                len(voted_segments) > 0 and \
                len(voted_selected_segments) > 0 and \
                voted_segments[-1][2] == voted_selected_segments[0][2] and \
                np.abs( voted_segments[-1][1] - voted_selected_segments[0][0] ) < 1e-6:
                voted_segments[-1][1] = voted_selected_segments[0][1]
                voted_selected_segments = voted_selected_segments[1:]
            
            voted_segments += voted_selected_segments
            prev_is_cutoff = is_cutoff
            
        if min_segment_length is not None:
            voted_segments = [ item for item in voted_segments if item[1] - item[0] >= min_segment_length  ]

            
        if len(voted_segments) == 0:
            pred_onsets, pred_offsets, pred_clusters = [], [], []
        else:
            pred_onsets, pred_offsets, pred_clusters = list(zip(*voted_segments  ))
        
        return {"onset":list(pred_onsets), "offset":list(pred_offsets), "cluster":list(pred_clusters) }
    
