import os,sys,inspect
from tqdm import tqdm
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import shutil
import json

import argparse

from utils import *
from model import *
from datautils import *

from transformers import AdamW, get_linear_schedule_with_warmup

def train_iteration(batch):
    for key in batch:
        batch[key] = batch[key].to(device)
        
    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):   
        model_out = model( **batch )
        loss = model_out.loss.mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    # optimizer.step()
    scaler.update()
    
    """ 
    # normal version without float16 speedup
    optimizer.zero_grad()
    model_out = model( **batch )
    loss = model_out.loss.mean()
    loss.backward()    
    optimizer.step()
    """
    return loss.item()

def validate( audio_list, label_list, segmenter, sr, tolerance, num_trials, min_segment_length,
              voting_time_step, voting_precision, batch_size, max_length, target_cluster = None ):
    total_n_true_positive, total_n_positive_in_prediction, total_n_positive_in_label = 0,0,0
    
    for audio, label in tqdm(zip(audio_list, label_list), total = len(audio_list)):        
        prediction = segmenter.segment( audio, num_trials = num_trials, min_segment_length = min_segment_length,
                                        voting_time_step = voting_time_step, voting_precision = voting_precision,
                                        batch_size = batch_size, max_length = max_length                                       
                                      )
        n_true_positive, n_positive_in_prediction, n_positive_in_label = segmenter.score( prediction, label, 
                                                                                          target_cluster = target_cluster,
                                                                                          tolerance = tolerance )[:3]
        
        total_n_true_positive += n_true_positive
        total_n_positive_in_prediction += n_positive_in_prediction
        total_n_positive_in_label += n_positive_in_label
    
    precision = total_n_true_positive / max(total_n_positive_in_prediction, 1e-12)
    recall = total_n_true_positive / max( total_n_positive_in_label, 1e-12 )
    f1 = 2/(1/max(precision, 1e-12) + 1/max(recall, 1e-12)  )
    return precision, recall, f1, total_n_true_positive, total_n_positive_in_prediction, total_n_positive_in_label

# class Args(object):
#     pass

# args = Args()

# args.timestamp_precision = 0.005
# args.timestamp_format = "<|%.3f|>"
# args.sr = 16000
# args.hop_length = None
# args.clip_duration = None
# args.max_length = 100
# args.batch_size = 4
# args.learning_rate = 1e-5
# args.lr_schedule = "linear"
# args.max_to_keep = -1
# args.seed = 66100
# args.tolerance = 0.02
# args.num_trials = 3
# args.min_segment_length = 0.02
# args.voting_time_step = 1.0
# args.voting_precision = 0.001
# args.weight_decay = 0.01
# args.warmup_steps = 100
# args.freeze_encoder = 0
# args.dropout = 0.0
# args.print_every = 100
# args.validate_every = None
# args.validate_per_epoch = 0
# args.save_every = 1000
# args.save_per_epoch = 0
# args.max_num_epochs = None
# args.max_num_iterations = 1000
# args.val_ratio = 0.0
# args.n_device = 1
# args.gpu_list = None
# args.initial_model_path = "openai/whisper-large"
# args.model_folder = "model/DAS/zebra_finch/whisper-large"
# args.result_folder = "result/DAS/zebra_finch/whisper-large"
# args.train_dataset_folder = "data/dataset/DAS/zebra_finch/train/"
# args.test_dataset_folder = "data/dataset/DAS/zebra_finch/test/"

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-timestamp_precision", type = float, default = 0.005 )
    parser.add_argument("-timestamp_format", default = "<|%.3f|>" )
    parser.add_argument("-sr", type = int, default = 16000 )
    parser.add_argument("-hop_length", type = int, default = None )
    parser.add_argument("-clip_duration", type = float, default = None )
    parser.add_argument("-max_length", type = int, default = 100 )
    parser.add_argument("-batch_size", type = int, default = 4 )
    parser.add_argument("-learning_rate", type = float, default = 1e-5 )
    parser.add_argument("-lr_schedule", default = "linear" )
    parser.add_argument("-max_to_keep", type = int, default = -1 )
    parser.add_argument("-seed", type = int, default = 66100 )
    parser.add_argument("-tolerance", type = float, default = 0.02 )
    parser.add_argument("-num_trials", type = int, default = 3 )
    parser.add_argument("-min_segment_length", type = float, default = 0.02 )
    parser.add_argument("-voting_time_step", type = float, default = 1.0 )
    parser.add_argument("-voting_precision", type = float, default = 0.001 )
    parser.add_argument("-weight_decay", type = float, default = 0.01 )
    parser.add_argument("-warmup_steps", type = int, default = 100 )
    parser.add_argument("-freeze_encoder", type = int, default = 0 )
    parser.add_argument("-dropout", type = float, default = 0.0 )
    parser.add_argument("-print_every", type = int, default = 100 )
    parser.add_argument("-validate_every", type = int, default = None )
    parser.add_argument("-validate_per_epoch", type = int, default = 0 )
    parser.add_argument("-save_every", type = int, default = 1500 )
    parser.add_argument("-save_per_epoch", type = int, default = 0 )
    parser.add_argument("-max_num_epochs", type = int, default = None )
    parser.add_argument("-max_num_iterations", type = int, default = 1500 )
    parser.add_argument("-val_ratio", type = float, default = 0.0 )
    parser.add_argument("-n_device", type = int, default = 1 )
    parser.add_argument("-gpu_list", type = int, nargs = "+", default = None )
    parser.add_argument("-num_workers", type = int, default = 4 )
    parser.add_argument("-clear_cluster_codebook", type = int, help="set the pretrained model's cluster_codebook to empty dict. This is used when we train the segmenter on a complete new dataset. Set this to 0 if you just want to slighlt finetune the model with some additional data with the same cluster naming rule.", default = 1 )
    
    parser.add_argument("-initial_model_path" )
    parser.add_argument("-model_folder" )
    parser.add_argument("-result_folder" )
    parser.add_argument("-train_dataset_folder" )
    parser.add_argument("-test_dataset_folder" )

    args = parser.parse_args()

    if args.hop_length is None:
        args.hop_length = int( args.sr * args.timestamp_precision * 0.5 ) # This 0.5 is due to the 2x downsampling in the conv layer
    if args.clip_duration is None:
        args.clip_duration = min( args.timestamp_precision * 500, 30.0)

    if args.seed is not None:
        np.random.seed(args.seed)  
        
    if args.val_ratio == 0.0:
        args.validate_every = None
        args.validate_per_epoch= None
    
    create_if_not_exists(args.model_folder)
    create_if_not_exists(args.result_folder)
    input_features_length = int( np.round( args.clip_duration / ( args.hop_length / args.sr ) ) )
    
    if args.gpu_list is None:
        args.gpu_list = np.arange(args.n_device).tolist()
    
    
    device = torch.device(  "cuda:%d"%( args.gpu_list[0] ) if torch.cuda.is_available() else "cpu" )
    
    model, feature_extractor, tokenizer, current_batch = load_model( args.model_folder, args.initial_model_path,  args.sr, 
                                                                     args.hop_length, input_features_length, args.timestamp_format, 
                                                                     args.timestamp_precision, args.clip_duration, args.dropout )
    model = model.to(device)
    
    if args.freeze_encoder:
        for para in model.model.encoder.parameters():
            para.requires_grad = False
    else:
        for para in model.model.encoder.parameters():
            para.requires_grad = True
            
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate )
    
    model = nn.DataParallel( model, args.gpu_list )
    
    
    segmenter = WhisperSegmenter( model = model, feature_extractor = feature_extractor, tokenizer = tokenizer )
    
    if args.clear_cluster_codebook:
        segmenter.update_cluster_codebook( {} )
    
    scaler = torch.cuda.amp.GradScaler()
    
    audio_path_list_train, label_path_list_train = get_audio_and_label_paths( args.train_dataset_folder )  
    audio_path_list_test, label_path_list_test = get_audio_and_label_paths( args.test_dataset_folder )  
    
    cluster_codebook = get_cluster_codebook( label_path_list_train + label_path_list_test, segmenter.cluster_codebook )
    segmenter.update_cluster_codebook( cluster_codebook )
    
    audio_list_train, label_list_train = load_data(audio_path_list_train, label_path_list_train, sr = args.sr, cluster_codebook = cluster_codebook, n_threads = 20 )  
    
    if args.val_ratio > 0:
        (audio_list_train, label_list_train), ( audio_list_val, label_list_val ) = train_val_split( audio_list_train, label_list_train, args.val_ratio, args.sr, feature_extractor.n_fft )
    audio_list_train, label_list_train = slice_audios_and_labels( audio_list_train, label_list_train, args.clip_duration, args.sr, feature_extractor.n_fft )
    
    audio_list_test, label_list_test = load_data(audio_path_list_test, label_path_list_test, sr = args.sr, cluster_codebook = cluster_codebook, n_threads = 20 )
    
    ## for the testing set and validation set, we get the original cluster name of each segment, because the segmenter's output is also cluster name not cluster_id,
    ## since cluster_id can be less interpretable
    for item in label_list_test:
        item["cluster"] = [ segmenter.inverse_cluster_codebook[cluster_id] for cluster_id in item["cluster_id"] ]
        
    if args.val_ratio > 0:
        for item in label_list_val:
            item["cluster"] = [ segmenter.inverse_cluster_codebook[cluster_id] for cluster_id in item["cluster_id"] ]
            
    training_dataset = VocalSegDataset( audio_list_train, label_list_train, feature_extractor, tokenizer, args.max_length, 
                       args.sr, args.clip_duration, input_features_length, args.timestamp_precision, args.timestamp_format  )
    
    training_dataloader = DataLoader( training_dataset, batch_size = args.batch_size , shuffle = True , 
                                             worker_init_fn = lambda x:[np.random.seed( int( time.time() )  + x ),  
                                                                    torch.manual_seed(int( time.time() ) + x) ] , 
                                             num_workers = args.num_workers , drop_last= True,
                                             pin_memory = False
                                           )
    
    if len(training_dataloader) == 0:
        print("Error: Too few examples (less than a batch) for training! Exit!")
        sys.exit(1)
    
    if args.max_num_epochs is not None and args.max_num_epochs > 0:
        args.max_num_iterations = len( training_dataloader ) * args.max_num_epochs
    else:
        assert args.max_num_iterations is not None and args.max_num_iterations > 0
        args.max_num_epochs = int(np.ceil( args.max_num_iterations / len( training_dataloader )  ))
                
    if args.lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps= args.warmup_steps, 
            num_training_steps = args.max_num_iterations
        )
    else:
        scheduler = None
        
    model.train() 
    training_loss_value_list = []
    for epoch in range(args.max_num_epochs):
        for count, batch in enumerate( tqdm( training_dataloader ) ):
            training_loss_value_list.append( train_iteration(batch) )
            
            if scheduler is not None:
                scheduler.step()
                
            current_batch += 1

            if current_batch % args.print_every == 0:
                print("Epoch: %d, current_batch: %d, learning rate: %f, Loss: %.4f"%( epoch, current_batch, get_lr(optimizer)[0], np.mean(training_loss_value_list)) )
                training_loss_value_list = [] 
                
            if  ( args.validate_every is not None and current_batch % args.validate_every == 0 ) or \
                ( args.validate_per_epoch and count == len(training_dataloader) - 1 ):
                print("Start validation ...")
                model.eval()
                ## in the validation set, set the num_trails to 1
                precision, recall, f1, TP, P_pred, P_true = validate( audio_list_val, label_list_val, segmenter, 
                                                                      args.sr, args.tolerance, args.num_trials, args.min_segment_length,
                                                                      args.voting_time_step, args.voting_precision, 
                                                                      args.batch_size, args.max_length
                                                                    )
                                
                print("Epoch: %d, current_batch: %d, validation syllable F1 score: %.2f"%( epoch, current_batch, f1 ))
                with open( args.result_folder+"/val_results.jsonl", "a") as f:
                    f.write( json.dumps( {
                        "current_batch":current_batch,
                        "validation_score":f1
                    } ) + "\n" )
                model.train()
                
            if ( args.save_every is not None and current_batch % args.save_every == 0 ) or \
               ( args.save_per_epoch and count == len(training_dataloader) - 1 ):
                model.eval()
                save_model( model, feature_extractor, tokenizer, current_batch, args.model_folder, args.max_to_keep )
                model.train()
                
            if current_batch >= args.max_num_iterations:
                break
        
        if current_batch >= args.max_num_iterations:
            break   
            
    try:
        assert args.val_ratio > 0
        all_validation_results = []
        with open( args.result_folder+"/val_results.jsonl", "r" ) as f:
            for line in f:
                line_data = json.loads(line)
                all_validation_results.append(( line_data["current_batch"], line_data["validation_score"] ))
        assert len(all_validation_results) > 0
        
        best_val_score = np.max( [ val_score for batch_num, val_score in all_validation_results ] )
        candidate_ckpt_and_smoothed_score_list = []
        for pos in range( len(all_validation_results) ):
            if np.abs( all_validation_results[pos][1] - best_val_score ) < 0.001:
                avg_neighboring_score = np.mean( [ val_score for batch_num, val_score in all_validation_results[max(pos-1,0):pos+2] ] )
                candidate_ckpt_and_smoothed_score_list.append( ( all_validation_results[pos][0], avg_neighboring_score ) )
        candidate_ckpt_and_smoothed_score_list.sort( key = lambda x:-x[1] )
        best_checkpoint_batch_number = candidate_ckpt_and_smoothed_score_list[0][0]
    except:
        best_checkpoint_batch_number = None
        
    ## if no validation results are availavle, load the last saved checkpoint as the best model
    if best_checkpoint_batch_number is None:  
        ckpt_list = glob( args.model_folder + "/*" )
        if len( ckpt_list ) >0:
            ckpt_list.sort( key = os.path.getmtime )
            ckpt_name = ckpt_list[-1]
            best_checkpoint_batch_number = int(ckpt_name.split("-")[-1])
            
            
    if best_checkpoint_batch_number is not None:
        print("The best checkpoint on validation set is: %s," % ( args.model_folder+"/checkpoint-%d"%(best_checkpoint_batch_number) ) )
        print("Reporting test results ...")
        segmenter = WhisperSegmenter( args.model_folder + "/checkpoint-%d"%( best_checkpoint_batch_number ), device )
        precision, recall, f1, TP, P_pred, P_true = validate( audio_list_test, label_list_test, segmenter, 
                                                              args.sr, args.tolerance, args.num_trials, args.min_segment_length,
                                                              args.voting_time_step, args.voting_precision, 
                                                              args.batch_size, args.max_length
                                                            )    
        
        with open( args.result_folder + "/test_results.txt", "w" ) as f:
            f.write("checkpoint-%d"%(best_checkpoint_batch_number) + "\n")
            f.write("f1 score: %.4f"%(f1) + "\n")
        print("Test performance: f1 score: %.4f"%(f1))
    
        print("Removing sub-optimal checkpoints ...")
        for ckpt_name in os.listdir( args.model_folder ):
            if ckpt_name != "checkpoint-%d"%(best_checkpoint_batch_number):
                shutil.rmtree( args.model_folder + "/" + ckpt_name )
    
    print("All Done!")    
    
    
