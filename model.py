import os
import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, WhisperConfig
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
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from audio_utils import WhisperSegFeatureExtractor
import time
from PIL import Image
from scipy.stats import mode

from utils import RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP

from huggingface_hub import snapshot_download
import hashlib
import os
import shutil
import threading

def download_model( model_path, ignore_cache = False ):
    ## This model path is a local folder path
    if os.path.exists( model_path ):
        return model_path
    ## Suppose that this model path is a model name stored at huggingface 
    cache_dir = os.path.expanduser(os.getenv("WHISPERSEG_MODEL_CACHE", "~/.cache/whisperseg_models/"))

    os.makedirs(cache_dir, exist_ok=True )
    
    model_folder_name = hashlib.sha256( model_path.encode()).hexdigest() 

    local_model_path = os.path.join( cache_dir, model_folder_name )  
    if ignore_cache:
        if os.path.exists( local_model_path ):
            shutil.rmtree( local_model_path )

    if not os.path.exists(local_model_path) or len(os.listdir(local_model_path)) == 0:
        os.makedirs(local_model_path, exist_ok=True )
        snapshot_download(model_path, local_dir = local_model_path )
    return local_model_path


def save_model( model, tokenizer, current_step, model_folder, max_to_keep ):
    try:
        model = model.module
    except:
        pass
    
    ckpt_list =  glob( model_folder + "/*" )
    tokenizer.save_pretrained( model_folder+"/checkpoint-%d"%(current_step) )

    model.config.current_step = current_step
    model.save_pretrained( model_folder+"/checkpoint-%d"%(current_step) )
    
    if max_to_keep > 0 and len( ckpt_list ) > max_to_keep:
        ckpt_list.sort( key = os.path.getmtime )
        ckpt_name = ckpt_list[0]
        os.system("rm -r %s"%(ckpt_name) )
        
def load_model( initial_model_path, total_spec_columns, dropout = 0.0):

    model = WhisperForConditionalGeneration.from_pretrained(initial_model_path)
    model.config.max_source_positions = int( 0.5*total_spec_columns )
    with torch.no_grad():
        model.model.encoder.embed_positions.weight = torch.nn.Parameter(
                        model.model.encoder.embed_positions.weight[:model.config.max_source_positions,:]
        )
        model.model.encoder.embed_positions.num_embeddings = model.config.max_source_positions
    
    model.config.total_spec_columns = total_spec_columns
    model.config.dropout = dropout
    model.model.encoder.dropout = dropout
    for layer in model.model.encoder.layers:
        layer.dropout = dropout
    model.model.decoder.dropout = dropout
    for layer in model.model.decoder.layers:
        layer.dropout = dropout

    if not hasattr( model.config, "cluster_codebook" ):
        model.config.cluster_codebook = {}

    if not hasattr( model.config, "species_codebook" ):
        model.config.species_codebook = {
            "zebra_finch":"<|zebra_finch|>",
            "bengalese_finch":"<|bengalese_finch|>",
            "mouse":"<|mouse|>",
            "marmoset":"<|marmoset|>",
            "human":"<|human|>",
            ## set unknown for other species
            "unknown":"<|unknown|>",
            "animal":"<|animal|>"
        }

    ## do not change nccratliri/whisper-large to openai/whisper-large, since the tokenizer in openai/whisper-large has changed its vocabulary
    tokenizer = WhisperTokenizer.from_pretrained("nccratliri/whisper-large", language = "english" )
    tokenizer.add_tokens( ["<|%d|>"%(i) for i in range( total_spec_columns + 1)], special_tokens=True  )
    tokenizer.add_tokens( [ v for k, v in model.config.species_codebook.items() ], special_tokens=True )
        
    return model, tokenizer

        
class SegmenterBase:
    def __init__( self,  ):
        self.segment_matcher = re.compile("<\|([0-9]+)\|>(\d+?)<\|([0-9]+)\|>")
        self.total_spec_columns = None
        self.precision_bits = 3
        self.cluster_codebook = None

    ### segmentation-related functions:
    def get_sliced_audios_features( self,  audio, sr, min_frequency, spec_time_step, num_trials):
        feature_extractor = WhisperSegFeatureExtractor( sr, spec_time_step, min_frequency = min_frequency )
        clip_duration = self.total_spec_columns * spec_time_step
        
        max_num_padding_samples = int( clip_duration * sr )
        audio_left_pad = np.zeros( ( audio.shape[0], max_num_padding_samples), dtype = np.float32 )
        audio_clip_length = int(clip_duration * sr)
        
        sliced_audios_features = []
        for trial_id in range(num_trials):            
            
            padding_time = np.round( clip_duration * trial_id / num_trials / spec_time_step ) * spec_time_step
            num_padding_samples = int( padding_time * sr )
            audio_padded = np.concatenate( [ audio_left_pad[:, audio_left_pad.shape[1] - num_padding_samples: ],
                                             audio
                                           ], axis = 1 
                                         )
            
            ## This loop must be executed once even for zero length audio
            for pos in range( 0, max(audio_padded.shape[1], 1), audio_clip_length ):
                offset_time = pos / sr - padding_time
                
                audio_clip = audio_padded[:,pos:pos+audio_clip_length]
                audio_clip_padded = np.concatenate([ audio_clip, np.zeros( ( audio_clip.shape[0], audio_clip_length - audio_clip.shape[1]), dtype = np.float32 ) ], axis = 1 )
            
                all_input_features = []
                for audio_clip_idx in range( audio_clip_padded.shape[0] ):
                    
                    input_features = feature_extractor(audio_clip_padded[audio_clip_idx], 
                                                       sampling_rate = sr, padding = "do_not_pad")["input_features"][0]
                    input_features = input_features[:,:self.total_spec_columns]
                    
                    if input_features.shape[1] > 0:
                        min_spec_value = input_features.min()
                    else:
                        min_spec_value = 0
                    # if input_features.shape[1] < self.total_spec_columns:
                    input_features = np.concatenate( [ input_features, 
                                                        min_spec_value * np.ones( ( input_features.shape[0], self.total_spec_columns - input_features.shape[1] ) )  ], axis = 1 ).astype(np.float32)
                    assert input_features.shape[1] == self.total_spec_columns
                    all_input_features.append( input_features )

                sliced_audios_features.append( ( trial_id, offset_time, all_input_features, audio_clip.shape[1]/sr ) )
        return sliced_audios_features
        
    ## This is used for WhisperSegmenter and WhisperSegmenterFast, not for WhisperSegmenterForEval
    def generate_segment_text( self, sliced_audios_features, batch_size, max_length, num_beams, top_k = 1, top_p = 1.0, length_penalty = 1.0, status_monitor = None ):
        generated_texts_dict = {}
        all_threads = []
        num_examples_per_thread = int(np.ceil( len( sliced_audios_features ) / len( self.device_list ) ))
        for thread_id, pos in enumerate(range( 0, len( sliced_audios_features ), num_examples_per_thread )):
            t = threading.Thread( target = self.generate_segment_text_core, 
                                  args = ( sliced_audios_features[pos:pos+num_examples_per_thread],
                                           batch_size, max_length, num_beams, top_k, top_p, 
                                           length_penalty, generated_texts_dict, thread_id,
                                           status_monitor if thread_id == 0 else None   ## pass the status_monitor only to the first thread
                                         )
                                )
            t.start()
            all_threads.append(t)
        for t in all_threads:
            t.join()
            
        generated_text_list = []
        for thread_id in sorted( list( generated_texts_dict.keys() ) ):
            generated_text_list += generated_texts_dict[thread_id]
        return generated_text_list  
    
    def extract_segments( self, text, spec_time_step ):
        inverse_cluster_codebook = { v:k for k,v in self.cluster_codebook.items()}   
        segment_list = []
        match_res_list = self.segment_matcher.findall( text )
        for onset_text, cluster_id_text, offset_text in match_res_list:
            onset = int( onset_text ) * spec_time_step * RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP
            offset = int( offset_text ) * spec_time_step * RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP
            cluster_id = int( cluster_id_text )
            
            if cluster_id not in inverse_cluster_codebook:
                continue
            if offset - onset <= 0:
                continue

            cluster = inverse_cluster_codebook[cluster_id]
            segment_list.append( [ onset, offset, cluster ] )
        return segment_list
    
    
    def parse_generation( self, generated_text_list, sliced_audios_features,
                         min_segment_length, 
                         audio_duration,
                         spec_time_step,
                         num_trials,
                         eps, time_per_frame_for_voting,
                         consolidation_method
                        ):
        
        ## convert generated text to on_offsets
        on_offset_list_of_trial = {}
        for count, generated_text in enumerate(generated_text_list):
            
            trial_id, offset_time, _, duration = sliced_audios_features[count]

            if trial_id not in on_offset_list_of_trial:
                on_offset_list_of_trial[trial_id] = []            

            on_offsets_clip = self.extract_segments( generated_text, spec_time_step )
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
                    ## previous offset == current onset and the cluster type is the same, then we merge them
                    
                    merged_on_offset_list_of_trial[trial_id][-1][1] = on_offsets_clip[0][1]
                    on_offsets_clip = on_offsets_clip[1:]
                merged_on_offset_list_of_trial[trial_id] += on_offsets_clip
        
        
        trials_results = []
        for trial_id in merged_on_offset_list_of_trial:
            for item in merged_on_offset_list_of_trial[trial_id]:
                item[0] = max( 0, item[0] )
                item[1] = min( item[1], audio_duration )
                
            merged_on_offset_list_of_trial[trial_id] = sorted( merged_on_offset_list_of_trial[trial_id], key = lambda x:x[0] )
            merged_on_offset_list_of_trial[trial_id] = [ item for item in merged_on_offset_list_of_trial[trial_id] if item[1] - item[0] >= min_segment_length ]
        
            pred_onsets, pred_offsets, pred_clusters = [], [], []
            for item in merged_on_offset_list_of_trial[trial_id]:            
                pred_onsets.append(item[0])
                pred_offsets.append(item[1])
                pred_clusters.append(item[2])
        
            trials_results.append(  {"onset":list(pred_onsets), "offset":list(pred_offsets), "cluster":list(pred_clusters) } )
        
        if num_trials == 1:
            final_prediction = trials_results[0]
        else:
            if consolidation_method == "clustering":
                min_samples = max( 2, int(np.ceil( num_trials * 0.5 )) )
                final_prediction = self.consolidate_trials_by_clustering( trials_results, eps, min_samples)
            else:
                final_prediction = self.consolidate_trials_by_voting( trials_results, time_per_frame_for_voting)
            
        ##formating the final prediction
        final_prediction["onset"] = [ float(np.round(t, self.precision_bits)) for t in final_prediction["onset"] ]
        final_prediction["offset"] = [ float(np.round(t, self.precision_bits)) for t in final_prediction["offset"] ]
        
        return final_prediction


    ### multi-trial consolidation 
    def custom_distance(self, segment1, segment2):
        onset_diff = abs(segment1[0] - segment2[0])
        offset_diff = abs(segment1[1] - segment2[1])
        return (onset_diff + offset_diff) / 2
    

    def consolidate_trials_by_clustering(self, trials, eps, min_samples):
        # Step 1: Create a list of all segments across all trials
        segments = []
        for trial_id, trial in enumerate(trials):
            for onset, offset, cluster in zip(trial['onset'], trial['offset'], trial['cluster']):
                segments.append({'onset': onset, 'offset': offset, 'cluster': cluster, 'trial': trial_id})
        if len(segments) == 0:
            return  {
                        "onset":[],
                        "offset":[],
                        "cluster": []
                    }

        # Step 2: Compute pairwise distance matrix
        dist_matrix = pairwise_distances([[seg['onset'], seg['offset']] for seg in segments], metric=self.custom_distance)

        # Step 3: Cluster segments using DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = db.fit_predict(dist_matrix)

        # Step 4: Merge segments within each cluster
        merged_segments = []
        for label in set(labels):
            if label != -1:  # Ignore noise
                group_segments = [seg for seg, lbl in zip(segments, labels) if lbl == label]
                if len(group_segments) == 0:
                    continue
                    
                cluster_name_dict = {}
                for seg in group_segments:
                    cluster_name_dict[ seg["cluster"] ] = cluster_name_dict.get( seg["cluster"], 0 ) + 1
                ## get the most common cluster name
                cluster_name = sorted(list(cluster_name_dict.items()), key = lambda x:-x[1])[0][0]
                
                avg_onset = np.mean([seg['onset'] for seg in group_segments])
                avg_offset = np.mean([seg['offset'] for seg in group_segments])
                merged_segments.append({'onset': avg_onset, 'offset': avg_offset, 'cluster': cluster_name })
                
        merged_segments.sort( key = lambda x:x["onset"] )
                        
        final_pred = {
            "onset":[ item["onset"] for item in merged_segments ],
            "offset":[ item["offset"] for item in merged_segments  ],
            "cluster": [ item["cluster"] for item in merged_segments ]
        }

        return final_pred
    
    def consolidate_trials_by_voting(self, trials, time_per_frame_for_voting):
        all_timestamps = []
        for trial in trials:
            all_timestamps += list( trial["onset"] )
            all_timestamps += list( trial["offset"] )
        if len(all_timestamps) == 0 or len(all_timestamps) % 2 != 0:
            return  {
                        "onset":[],
                        "offset":[],
                        "cluster": []
                    }
        min_time = np.min( all_timestamps )
        max_time = np.max( all_timestamps )
        num_frames = int(np.round( ( max_time - min_time ) / time_per_frame_for_voting )) 
        
        all_frame_wise_predictions = []
        for trial in trials:
            frame_wise_prediction =  np.ones( num_frames ) * -1
            for pos in range(len( trial["onset"] )):
                onset = trial["onset"][pos] - min_time
                offset = trial["offset"][pos] - min_time
                cluster_id = self.cluster_codebook[trial["cluster"][pos]]
                frame_wise_prediction[ int( np.round(onset / time_per_frame_for_voting) ): int( np.round( offset / time_per_frame_for_voting ) ) ] = cluster_id
            all_frame_wise_predictions.append( frame_wise_prediction )
            
        all_frame_wise_predictions = np.asarray(all_frame_wise_predictions)
        voted_frame_wise_prediction, counts = mode(all_frame_wise_predictions, axis=0)
        voted_frame_wise_prediction_right_pad = np.array( voted_frame_wise_prediction.tolist() + [-1] ) 
        voted_frame_wise_prediction_left_pad = np.array(  [-1] + voted_frame_wise_prediction.tolist() ) 
        event_positions = np.argwhere(voted_frame_wise_prediction_right_pad - voted_frame_wise_prediction_left_pad != 0)[:,0]
        
        final_onsets = []
        final_offsets = []
        final_clusters = []
        
        inverse_cluster_codebook = { v:k for k,v in self.cluster_codebook.items() } 
        
        for idx in range(0,len(event_positions)-1):
            onset_pos = event_positions[ idx ]
            offset_pos = event_positions[ idx + 1 ]
            cluster_id = int(np.round(np.mean(voted_frame_wise_prediction[ onset_pos: offset_pos ])))
            
            if cluster_id == -1:
                continue
            
            final_onsets.append( onset_pos * time_per_frame_for_voting + min_time )
            final_offsets.append( offset_pos * time_per_frame_for_voting + min_time )
            final_clusters.append( inverse_cluster_codebook[ cluster_id ] )
            
        final_pred = {
            "onset":final_onsets,
            "offset":final_offsets,
            "cluster":final_clusters
        }

        return final_pred
    
    
    @torch.no_grad()
    def segment( self, audio, sr,
                       min_frequency = 0,
                       spec_time_step = 0.0025,
                       min_segment_length = 0.01,
                       eps = 0.02,  ## for DBSCAN clustering
                       time_per_frame_for_voting = 0.001, ## for voting
                       consolidation_method = "clustering",
                       max_length = 448, 
                       batch_size = 4, 
                       num_trials = 3,
                       num_beams = 4,
                       top_k = 1, 
                       top_p = 1.0, 
                       length_penalty = 1.0,
                       status_monitor = None 
               ):
        tic1 = time.time()
        sliced_audios_features = self.get_sliced_audios_features( audio, sr, min_frequency, spec_time_step, num_trials)
        tic2 = time.time()
        generated_text_list = self.generate_segment_text( sliced_audios_features, batch_size, max_length, num_beams, top_k, top_p, length_penalty, status_monitor )
        tic3 = time.time()
        
        final_prediction = self.parse_generation( 
            generated_text_list, sliced_audios_features,
                         min_segment_length, 
                         audio.shape[1] /sr,
                         spec_time_step,
                         num_trials,
                         eps, time_per_frame_for_voting,
                         consolidation_method 
                        )
        tic4 = time.time()
        
        
        return final_prediction
            

    ### evaluation-related functions
    def compute_syllable_score( self, prediction_on_offset_list, label_on_offset_list, tolerance = 0.02  ):
        
        n_positive_in_prediction = len(prediction_on_offset_list)
        n_positive_in_label = len(label_on_offset_list)
        
        n_true_positive = 0
        for pred_onset, pred_offset in prediction_on_offset_list:
            is_matched = False
            for count, (label_onset, label_offset) in enumerate( label_on_offset_list ):
                if np.abs( pred_onset - label_onset )<=tolerance and np.abs( pred_offset - label_offset )<= tolerance:
                    n_true_positive += 1
                    is_matched = True
                    break  # early stop for the predicted value
            if is_matched:
                ## remove the already matched syllable from the ground-truth
                label_on_offset_list.pop(count)
        
        return n_true_positive, n_positive_in_prediction, n_positive_in_label

    def segment_score( self, prediction, label, target_cluster = None, tolerance = 0.02 ):
        
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
    
    def frame_score(self, prediction, label, target_cluster = None, time_per_frame_for_scoring = 0.01 ):
        prediction_segments = prediction
        label_segments = label
        
        prediction_segments["cluster"] = list( map(str, prediction_segments["cluster"]) )
        label_segments["cluster"] = list( map(str, label_segments["cluster"]) )
                
        cluster_to_id_mapper = {}
        for cluster in list(prediction_segments["cluster"]) + list(label_segments["cluster"]):
            if cluster not in cluster_to_id_mapper:
                cluster_to_id_mapper[cluster] = len( cluster_to_id_mapper )
        
        all_timestamps = list(prediction_segments["onset"]) + list(prediction_segments["offset"]) + \
                            list(label_segments["onset"]) + list( label_segments["offset"] )
        if len(all_timestamps) == 0:
            max_time = 1.0
        else:
            max_time = np.max( all_timestamps )
            
        num_frames = int(np.round( max_time / time_per_frame_for_scoring )) + 1
        
        frame_wise_prediction = np.ones( num_frames ) * -1
        for idx in range( len( prediction_segments["onset"] ) ):
            onset_pos = int(np.round( prediction_segments["onset"][idx] / time_per_frame_for_scoring ))
            offset_pos = int(np.round( prediction_segments["offset"][idx] / time_per_frame_for_scoring ))
            frame_wise_prediction[onset_pos:offset_pos] = cluster_to_id_mapper[ prediction_segments["cluster"][idx] ]
            
        frame_wise_label = np.ones( num_frames ) * -1
        for idx in range( len( label_segments["onset"] ) ):
            onset_pos = int(np.round( label_segments["onset"][idx] / time_per_frame_for_scoring ))
            offset_pos = int(np.round( label_segments["offset"][idx] / time_per_frame_for_scoring ))
            frame_wise_label[onset_pos:offset_pos] = cluster_to_id_mapper[ label_segments["cluster"][idx] ]
            
        if target_cluster is None:
            TP = np.logical_and( frame_wise_label != -1, frame_wise_prediction == frame_wise_label ).sum()
            P_in_pred = (frame_wise_prediction != -1).sum()
            P_in_label = (frame_wise_label != -1).sum()
        else:
            target_cluster_id = cluster_to_id_mapper[target_cluster]
            TP = np.logical_and( frame_wise_label == target_cluster_id, frame_wise_prediction == frame_wise_label ).sum()
            P_in_pred = (frame_wise_prediction == target_cluster_id).sum()
            P_in_label = (frame_wise_label == target_cluster_id).sum()
            
        
        precision = TP / max(P_in_pred, 1e-12)
        recall = TP / max(P_in_label, 1e-12)
        f1 = 2/( 1/max( precision, 1e-12 ) + 1/max( recall, 1e-12 ) )
                
        return TP, P_in_pred, P_in_label, precision, recall, f1
    
    
class WhisperSegmenterForEval(SegmenterBase):        
    def __init__(self, model_path = None, device = None, model = None, tokenizer = None):
        super().__init__()
        if model_path is not None:
            self.model = WhisperForConditionalGeneration.from_pretrained( model_path )
            self.tokenizer = WhisperTokenizer.from_pretrained(model_path, language = "english" )
            if device is not None:
                self.model = self.model.to(device)
        else:
            try:
                self.model = model.module
            except:
                self.model = model
            
            self.tokenizer = tokenizer
        
        self.device = list( self.model.parameters() )[0].device
        
        self.total_spec_columns = self.model.config.total_spec_columns
        self.cluster_codebook = self.model.config.cluster_codebook
        self.inverse_cluster_codebook = { cluster_id:cluster  for cluster, cluster_id in self.cluster_codebook.items() }
            
    def update_cluster_codebook(self, cluster_codebook):
        self.model.config.cluster_codebook = cluster_codebook
        
        self.cluster_codebook = cluster_codebook
        self.inverse_cluster_codebook = { cluster_id:cluster for cluster, cluster_id in self.cluster_codebook.items() }
                

    def generate_segment_text( self, sliced_audios_features, batch_size, max_length, num_beams, top_k = 1, top_p = 1.0, length_penalty = 1.0, status_monitor = None ):
        generated_text_list = []
        
        for pos in range( 0, len(sliced_audios_features), batch_size ):

            
            ### debugging here, merge the three spectrograms into one
            input_features = torch.from_numpy( np.asarray([ np.concatenate(item[2], axis = 0)[::len(item[2])]  for item in sliced_audios_features[pos:pos+batch_size] ]) ).to(self.device)

            
            generated_ids = self.model.generate( inputs = input_features,  
                                                 decoder_input_ids = torch.LongTensor([ self.tokenizer.convert_tokens_to_ids( [ "<|startoftranscript|>", "<|en|>", "<|notimestamps|>"] ) for _ in range( input_features.size(0) )]).to(self.device),
                                                 pad_token_id = self.tokenizer.pad_token_id,
                                                 eos_token_id = self.tokenizer.eos_token_id,
                                                 max_length = max_length,
                                                 num_beams = num_beams,
                                                 do_sample = num_beams == 1,
                                                 top_k = top_k,
                                                 top_p = top_p,
                                                 length_penalty = length_penalty
                                               )
            generated_text_batch = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            generated_text_list += generated_text_batch
        return generated_text_list
    
    
class WhisperSegmenter(SegmenterBase):        
    def __init__(self, model_path, device = None, device_ids = [ 0,] ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if device == "cpu":
            self.device_list = [ torch.device("cpu") ]
            self.model_list = [ WhisperForConditionalGeneration.from_pretrained( model_path ) ]
        else:
            self.device_list = [ torch.device("cuda", gpu) for gpu in device_ids ]
            self.model_list = [ WhisperForConditionalGeneration.from_pretrained( model_path ).to(device) for device in self.device_list ]
        self.tokenizer_list = [ WhisperTokenizer.from_pretrained(model_path, language = "english" ) for _ in self.device_list ]
        
        self.total_spec_columns = self.model_list[0].config.total_spec_columns
        self.cluster_codebook = self.model_list[0].config.cluster_codebook
        self.inverse_cluster_codebook = { cluster_id:cluster  for cluster, cluster_id in self.cluster_codebook.items() }         
        
        
    def generate_segment_text_core( self, sliced_audios_features, batch_size, max_length, num_beams, top_k, top_p, 
                                          length_penalty,  generated_texts_dict, thread_id, status_monitor = None ):
        device = self.device_list[thread_id]
        model = self.model_list[thread_id]
        tokenizer = self.tokenizer_list[thread_id]
        generated_text_list = []
        for pos in range( 0, len(sliced_audios_features), batch_size ):

            ### debugging here, merge the three spectrograms into one
            input_features = torch.from_numpy( np.asarray([ np.concatenate(item[2], axis = 0)[::len(item[2])] for item in sliced_audios_features[pos:pos+batch_size] ]) ).to(device)

            
            generated_ids = model.generate( inputs = input_features,  
                                                 decoder_input_ids = torch.LongTensor([ tokenizer.convert_tokens_to_ids( [ "<|startoftranscript|>", "<|en|>", "<|notimestamps|>"] ) 
                                                                                          for _ in range( input_features.size(0) )]).to(device),
                                                 pad_token_id = tokenizer.pad_token_id,
                                                 eos_token_id = tokenizer.eos_token_id,
                                                 max_length = max_length,
                                                 num_beams = num_beams,
                                                 do_sample = num_beams == 1,
                                                 top_k = top_k,
                                                 top_p = top_p,
                                                 length_penalty = length_penalty
                                               )
            generated_text_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            generated_text_list += generated_text_batch
            
            ### if status_monitor is not None, update the progress in percentage
            ### status_monitor is a dictionary which contains the key "progress" 
            if status_monitor is not None:
                progress_percent =  int( 100 * min( 1, (pos+batch_size) / len(sliced_audios_features) ) )
                status_monitor["progress"] = progress_percent
            
        generated_texts_dict[thread_id] = generated_text_list
    
class WhisperSegmenterFast(SegmenterBase):
    def __init__(self, model_path, device=None, device_ids = [ 0,] ):
        super().__init__()
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = download_model( model_path, ignore_cache = False )
            
        if device == "cpu":
            compute_type = "float32"
            self.device_list = [ 0,]
            self.model_list = [ ctranslate2.models.Whisper(model_path, device = device, compute_type = compute_type) ]
        else:
            compute_type = "float16"
            self.device_list = device_ids
            self.model_list = [ ctranslate2.models.Whisper(model_path, device = device, device_index = idx, compute_type = compute_type) for idx in device_ids ]
        self.tokenizer_list = [ WhisperTokenizer.from_pretrained(model_path+"/hf_model", language = "english" ) for _ in self.device_list ]
        
        model_config = json.load(open(model_path+"/hf_model"+"/config.json"))        
        self.total_spec_columns = model_config["total_spec_columns"]
        self.cluster_codebook = model_config["cluster_codebook"]
        self.inverse_cluster_codebook = { cluster_id:cluster  for cluster, cluster_id in self.cluster_codebook.items() }

    def generate_segment_text_core( self, sliced_audios_features, batch_size, max_length, num_beams, top_k, top_p, 
                                          length_penalty, generated_texts_dict, thread_id, status_monitor = None ):
        
        tokenizer = self.tokenizer_list[thread_id]
        model = self.model_list[thread_id]
        generated_text_list = []
        for pos in range( 0, len(sliced_audios_features), batch_size ):
            
            """ 
            This is the code if model is the converted ctranslate model
            """
            sliced_audios_features_batch = sliced_audios_features[pos:pos+batch_size]
            actual_batch_size = len(sliced_audios_features_batch)

            ### debugging here, merge the three spectrograms into one
            features = ctranslate2.StorageView.from_array(np.asarray([ np.concatenate(item[2], axis = 0)[::len(item[2])] for item in sliced_audios_features_batch ]))
            
            prompt = tokenizer.convert_tokens_to_ids(
                [ "<|startoftranscript|>", "<|en|>", "<|notimestamps|>"]
            )
            ## the ctranslate converted model typically requires a larger max length than the one required by the original huggingface model, so we set max_length to a large value.
            ## Note Ctranslate Whisper does not support top_p sampling
            model_output = model.generate(features, [ prompt for _ in range(actual_batch_size) ], 
                                                 max_length = max_length, beam_size = num_beams,
                                                 sampling_topk = top_k,
                                                 length_penalty = length_penalty
                                              )
            generated_text_batch = []
            for item in model_output:
                try:
                    gen_text = "".join(item.sequences[0])
                except:
                    gen_text = ""
                generated_text_batch.append(gen_text)
                
            generated_text_list += generated_text_batch
            
            
            ### if status_monitor is not None, update the progress in percentage
            ### status_monitor is a dictionary which contains the key "progress" 
            if status_monitor is not None:
                progress_percent =  int( 100 * min( 1, (pos+batch_size) / len(sliced_audios_features) ) )
                status_monitor["progress"] = progress_percent
            
        
        generated_texts_dict[thread_id] = generated_text_list
    

class MultiChannelWhisperSeg:
    def __init__(self, model_path, device=None, device_ids = [ 0,] ):
        try:
            segmenter = WhisperSegmenterFast(  model_path = model_path, device=device, device_ids = device_ids )
        except:
            segmenter = WhisperSegmenter(  model_path = model_path, device=device, device_ids = device_ids )
        self.segmenter = segmenter

    def print_progress_bar(self, status_monitor, title = "", length=50):
        while True:
            if status_monitor.get("finished", False):
                break
            percent = status_monitor.get("progress", 0)
            bar_length = int(length * percent / 100)
            bar = '■' * bar_length + '-' * (length - bar_length)
            print(f'{title} [{bar}] {percent:.2f}%', flush = False, end = "\r")
            time.sleep( 0.02 )
        print()
            
    def segment(self, radio_channels: list, mic_channels: list, sr: int,
                      min_frequency = 0,
                      spec_time_step = 0.0025,
                      min_segment_length = 0.005,
                      eps = 0.02,  ## for DBSCAN clustering
                      time_per_frame_for_voting = 0.001, ## for voting
                      consolidation_method = "clustering",
                      max_length = 448, 
                      batch_size = 4, 
                      num_trials = 3,
                      num_beams = 4,
                      top_k = 1, 
                      top_p = 1.0, 
                      length_penalty = 1.0,
               ):
        audio_len_list = []
        for item in radio_channels + mic_channels:
            assert len(item.shape) == 1, "Each audio channel must be one-dimensional!"
            audio_len_list.append(len(item))
        assert len(set(audio_len_list)) == 1, "All audio channels must have the same length!"
        assert len(radio_channels) > 0 and len(mic_channels) > 0

        predictions = []
        for channel_idx in range( len(radio_channels) ):
            target_radio = radio_channels[channel_idx]
            non_target_audio = np.zeros_like(target_radio)
            for idx in range( len(radio_channels) ):
                if idx != channel_idx:
                    non_target_audio += radio_channels[idx]
            mic_audio = mic_channels[0]
            audio = np.asarray( [ target_radio, non_target_audio, mic_audio ] )

            status_monitor = {}
            t = threading.Thread( target = self.print_progress_bar, args = [status_monitor, "Segmenting radio channel %d"%( channel_idx ) ] )
            t.start()
            prediction = self.segmenter.segment(  audio, sr = sr, min_frequency = min_frequency, spec_time_step = spec_time_step,
                       min_segment_length = min_segment_length, eps = eps, time_per_frame_for_voting = time_per_frame_for_voting,
                       consolidation_method = consolidation_method, max_length = max_length, batch_size=batch_size,
                       num_trials = num_trials, num_beams = num_beams, top_k = top_k, top_p = top_p, length_penalty = length_penalty,
                       status_monitor = status_monitor
                                          )
            status_monitor["finished"] = True
            t.join()
            predictions.append( prediction )
        return predictions