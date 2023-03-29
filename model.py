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

        feature_extractor = WhisperFeatureExtractor(  
            feature_size=80,
            sampling_rate=sr,
            hop_length=hop_length,
            chunk_length=30,
            n_fft=400,
            padding_value=0.0,
            return_attention_mask=False)
        tokenizer = WhisperTokenizer.from_pretrained(initial_model_path, language = "english" )
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
        pass
    
    def extract_on_offset( self, text, duration ):
        if text.startswith("IN"):
            text = "<|0.00|>" + text
        syllable_matcher = re.compile("<\|([0-9]+\.[0-9]+)\|>IN<\|([0-9]+\.[0-9]+)\|>")
        boundary_syllable_matcher = re.compile("<\|([0-9]+\.[0-9]+)\|>IN$")
        matched_pairs = syllable_matcher.findall(text)
        on_offset_list = []
        for onset_text, offset_text in matched_pairs:
            try:
                onset = float(onset_text)
                offset = float(offset_text)
                if offset - onset > 0:
                    on_offset_list.append( [onset, offset] )
            except:
                continue
        
        try:
            final_syllable_onset = float(boundary_syllable_matcher.findall(text)[-1])
            if duration - final_syllable_onset > 0:
                on_offset_list.append( [final_syllable_onset, duration ] )
        except:
            pass
        return on_offset_list
    
    def score( self, prediction, label, tolerance = 0.02 ):
        prediction_on_offset_list = []
        for pos in range(len(prediction["onset"])):
            prediction_on_offset_list.append([ prediction["onset"][pos], prediction["offset"][pos] ])
            
        label_on_offset_list = []
        for pos in range(len(label["onset"])):
            label_on_offset_list.append([ label["onset"][pos], label["offset"][pos] ])
        
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
    
    """"
    The following functions are used for implement an interactive visulization function to see the spectrogram and the label
    """

    def chunk_audio(self, audio, start_time, end_time, sr):
        start_idx = int( start_time * sr )
        end_idx = int( end_time * sr )
        chunked_audio = audio[start_idx:end_idx]
        return chunked_audio    

    def chunk_label(self, label, start_time, end_time ):
        intersected_indices = np.logical_and( label["onset"] < end_time, label["offset"] > start_time )
        chunked_label = {
                "onset": np.maximum(label["onset"][intersected_indices], start_time ) - start_time ,
                "offset":np.minimum(label["offset"][intersected_indices], end_time ) - start_time
            }
        return chunked_label    
    
    def compute_score(self, prediction, label, tolerance ):
        TP, n_positive_in_prediction, n_positive_in_label = self.score(  prediction=prediction, 
                                                                                           label=label, 
                                                                                           tolerance=tolerance )
        FP = n_positive_in_prediction - TP
        FN = n_positive_in_label - TP
        return {
            "TP":TP,
            "FP":FP,
            "FN":FN
        }   

    
    def plot_spec_and_labels(self, offset, window_size, audio, prediction, label, sr, tolerance, audio_file_name ):
        report_score = True
        if label is None:
            label = {"onset":np.array([]), "offset":np.array([])}
            report_score = False
        if prediction is None:
            prediction = {"onset":np.array([]), "offset":np.array([])}
            report_score = False
        
        start_time = offset
        end_time = start_time + window_size
        
        audio_chunked = self.chunk_audio( audio, start_time, end_time, sr )
        label_chunked = self.chunk_label( label, start_time, end_time )
        prediction_chunked = self.chunk_label( prediction, start_time, end_time )
        
        if report_score:
            score_of_current_chunk = self.compute_score( prediction_chunked, label_chunked, tolerance )
        
        spec = self.feature_extractor( audio_chunked, sampling_rate=sr, padding = "do_not_pad")["input_features"][0]
        
        time_per_column = self.feature_extractor.hop_length / sr
        spec_xticks_step_size = int(np.round( 0.5 / time_per_column )) 
        spec_xticks_values = np.arange(0, spec.shape[1]+1, spec_xticks_step_size )
        spec_xticks_labels = np.round(spec_xticks_values * time_per_column + start_time, 2) 
        
        spec_labels = np.zeros( spec.shape[1], dtype = np.int32 )
        for pos in range(len(label_chunked["onset"])):
            onset_idx = int(np.round(label_chunked["onset"][pos]/time_per_column))
            offset_idx = int(np.round(label_chunked["offset"][pos]/time_per_column))
            spec_labels[onset_idx:offset_idx] = 1
            
        spec_labels_image = np.ones( ( len(spec_labels), 3 ), dtype = np.float32 )
        spec_labels_image[ spec_labels == 1, :] = np.array([0.0, 0.0, 1.0])
        spec_labels_image = np.tile( spec_labels_image[np.newaxis,:,:], [40,1,1] )
            
        spec_preds = np.zeros( spec.shape[1], dtype = np.int32 )
        for pos in range(len(prediction_chunked["onset"])):
            onset_idx = int(np.round(prediction_chunked["onset"][pos]/time_per_column))
            offset_idx = int(np.round(prediction_chunked["offset"][pos]/time_per_column))
            spec_preds[onset_idx:offset_idx] = 1
            
        spec_preds_image = np.ones( (len(spec_preds), 3), dtype = np.float32 )
        spec_preds_image[ spec_preds == 1, : ] = np.array([1.0, 0.0, 0.0])
        spec_preds_image = np.tile( spec_preds_image[np.newaxis,:,:], [40,1,1] )
        
        canvas_image = np.ones( (100, len(spec_labels), 3 ) )
        canvas_image[:40,:,:] = spec_preds_image 
        canvas_image[50:90,:,:] = spec_labels_image
            
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,4), tight_layout=True, sharex=False)
        
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.set_xticks([])
        axes[1].spines['bottom'].set_visible(True)
        
        axes[0].imshow( spec, cmap="jet", origin="lower" )
        axes[0].text(-137,35,"Spectrogram:", fontfamily = "monospace" )
        axes[0].text(-137,90,"Wav file name: %s"%(audio_file_name), fontfamily = "monospace" )
        axes[1].imshow( canvas_image ) 
        axes[1].set_xticks( spec_xticks_values, spec_xticks_labels )
        axes[1].set_xlabel("time (s)")
        axes[1].text(-130,23,"Prediction:", fontfamily = "monospace" )
        axes[1].text(-130,78,"Label:", fontfamily = "monospace" )
        
        if report_score:
            axes[1].text(-130,150,"Error:  FP: %d, FN: %d"%( score_of_current_chunk["FP"], score_of_current_chunk["FN"] ), fontfamily = "monospace" )
        
        plt.subplots_adjust(wspace=0, hspace=-0.8)    
        plt.show()
        
    def visualize( self, audio, prediction = None, label = None, audio_file_name = "", tolerance = 0.02, window_size = 5 ):
        sr = self.sr
        return interact(self.plot_spec_and_labels, 
                    offset=(0, max(0, len(audio)/sr - window_size ), 0.1), 
                    window_size = fixed(window_size), 
                    audio = fixed(audio), 
                    prediction = fixed(prediction),
                    label = fixed(label), 
                    sr = fixed(sr), 
                    tolerance = fixed(tolerance), 
                    audio_file_name = fixed(audio_file_name))
    

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
        self.input_features_length = int(self.clip_duration/ (self.feature_extractor.hop_length / self.sr))
        
    @torch.no_grad()
    def segment( self, audio, max_length = 100 , num_trials = 3, batch_size = 16  ):
        
        clip_duration = self.clip_duration
        sr = self.sr
        timestamp_precision = self.timestamp_precision
        prediction_timesabstamps = np.zeros( int(len(audio) / sr / timestamp_precision) + 1 )    
        max_num_padding_samples = int( 1.0 * sr )
        audio_left_pad = np.zeros( max_num_padding_samples, dtype = np.float32 )
        audio_clip_length = int(clip_duration * sr)
        
        sliced_audios_features = []
        for trial_id in range(num_trials):            
            num_padding_samples = int( len(audio_left_pad) * trial_id / num_trials )
            padding_time = num_padding_samples / sr
            
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
            
            # This is the code if self.model is the converted ctranslate model
            sliced_audios_features_batch = sliced_audios_features[pos:pos+batch_size]
            actual_batch_size = len(sliced_audios_features_batch)
            features = ctranslate2.StorageView.from_array(np.asarray([ item[2] for item in sliced_audios_features_batch ]))
            prompt = self.tokenizer.convert_tokens_to_ids(
                [ "<|startoftranscript|>", "<|en|>", "<|notimestamps|>"]
            )
            ## the ctranslate converted model typically requires a larger max length than the one required by the original huggingface model, so we set max_length to a large value.
            model_output = self.model.generate(features, [ prompt for _ in range(actual_batch_size) ], max_length = max( 448, 2*max_length ) )
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

            on_offsets_clip = self.extract_on_offset( generated_text, duration )
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
                   merged_on_offset_list_of_trial[trial_id][-1][1] == on_offsets_clip[0][0]:
                    
                    merged_on_offset_list_of_trial[trial_id][-1][1] = on_offsets_clip[0][1]
                    on_offsets_clip = on_offsets_clip[1:]
                merged_on_offset_list_of_trial[trial_id] += on_offsets_clip
            
        
        for trial_id in merged_on_offset_list_of_trial:
            if len(merged_on_offset_list_of_trial[trial_id]) > 0:
                merged_onset_list, merged_offset_list = list( zip(*merged_on_offset_list_of_trial[trial_id]) )
            else:
                merged_onset_list, merged_offset_list = [], []
            pred = {
                "onset": np.array( merged_onset_list ),
                "offset": np.array( merged_offset_list )
            }
            for idx in range( len(pred["onset"]) ):
                onset, offset = pred["onset"][idx], pred["offset"][idx]
                onset = max(0, onset)
                offset = max(0, offset)
                
                prediction_timesabstamps[ int(np.round( onset / timestamp_precision)) : int(np.round( offset / timestamp_precision )) ] += 1
            
        
        prediction_timesabstamps /= num_trials
        prediction_timesabstamps = (prediction_timesabstamps > 0.5).astype(np.int32)
        
        padded_pred = np.concatenate([ np.array([0.0]), prediction_timesabstamps, np.array([0.0]) ], axis = 0 )
        
        pred_onsets = np.argwhere( padded_pred[1:] - padded_pred[:-1] > 0 )[:,0] * timestamp_precision
        pred_offsets = np.argwhere( padded_pred[1:] - padded_pred[:-1] < 0 )[:,0] * timestamp_precision
                
        return {"onset":np.round(pred_onsets,2), "offset":np.round(pred_offsets, 2) }
    
    
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
        self.input_features_length = int(self.clip_duration/ (self.feature_extractor.hop_length / self.sr))
        
        
    @torch.no_grad()
    def segment( self, audio, max_length = 100 , num_trials = 3, batch_size = 16  ):
        
        clip_duration = self.clip_duration
        sr = self.sr
        timestamp_precision = self.timestamp_precision
        prediction_timesabstamps = np.zeros( int(len(audio) / sr / timestamp_precision) + 1 )    
        max_num_padding_samples = int( 1.0 * sr )
        audio_left_pad = np.zeros( max_num_padding_samples, dtype = np.float32 )
        audio_clip_length = int(clip_duration * sr)
        
        sliced_audios_features = []
        for trial_id in range(num_trials):            
            num_padding_samples = int( len(audio_left_pad) * trial_id / num_trials )
            padding_time = num_padding_samples / sr
            
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

            on_offsets_clip = self.extract_on_offset( generated_text, duration )
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
                   merged_on_offset_list_of_trial[trial_id][-1][1] == on_offsets_clip[0][0]:
                    
                    merged_on_offset_list_of_trial[trial_id][-1][1] = on_offsets_clip[0][1]
                    on_offsets_clip = on_offsets_clip[1:]
                merged_on_offset_list_of_trial[trial_id] += on_offsets_clip
            
        
        for trial_id in merged_on_offset_list_of_trial:
            if len(merged_on_offset_list_of_trial[trial_id]) > 0:
                merged_onset_list, merged_offset_list = list( zip(*merged_on_offset_list_of_trial[trial_id]) )
            else:
                merged_onset_list, merged_offset_list = [], []
            pred = {
                "onset": np.array( merged_onset_list ),
                "offset": np.array( merged_offset_list )
            }
            for idx in range( len(pred["onset"]) ):
                onset, offset = pred["onset"][idx], pred["offset"][idx]
                onset = max(0, onset)
                offset = max(0, offset)
                
                prediction_timesabstamps[ int(np.round( onset / timestamp_precision)) : int(np.round( offset / timestamp_precision )) ] += 1
            
        
        prediction_timesabstamps /= num_trials
        prediction_timesabstamps = (prediction_timesabstamps > 0.5).astype(np.int32)
        
        padded_pred = np.concatenate([ np.array([0.0]), prediction_timesabstamps, np.array([0.0]) ], axis = 0 )
        
        pred_onsets = np.argwhere( padded_pred[1:] - padded_pred[:-1] > 0 )[:,0] * timestamp_precision
        pred_offsets = np.argwhere( padded_pred[1:] - padded_pred[:-1] < 0 )[:,0] * timestamp_precision
                
        return {"onset":np.round(pred_onsets,2), "offset":np.round(pred_offsets, 2) }
