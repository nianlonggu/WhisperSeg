import os,sys,inspect
script_dirname = os.path.dirname(os.path.abspath(__file__))
from tqdm import tqdm
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import shutil
import json
import wandb
import argparse
from audio_utils import SpecViewer, WhisperSegFeatureExtractor
from utils import *
from model import *
from datautils import *
from evaluate import evaluate
import subprocess
import json

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



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--initial_model_path" )
    parser.add_argument("--model_folder" )
    parser.add_argument("--train_dataset_folder" )
    parser.add_argument("--n_device", type = int, default = 1 )
    parser.add_argument("--gpu_list", type = int, nargs = "+", default = None )
    parser.add_argument("--use_wandb", type = int, default = 0 )
    parser.add_argument("--project", default = "whisperseg-multi-species" )
    parser.add_argument("--run_name", default = None )
    parser.add_argument("--print_every", type = int, default = 100 )
    parser.add_argument("--validate_every", type = int, default = None )
    parser.add_argument("--validate_per_epoch", type = int, default = 0 )
    parser.add_argument("--save_every", type = int, default = None )
    parser.add_argument("--save_per_epoch", type = int, default = 0 )
    parser.add_argument("--max_num_epochs", type = int, default = 3 )
    parser.add_argument("--max_num_iterations", type = int, default = None )
    parser.add_argument("--min_num_iterations", type = int, default = 500 )
    parser.add_argument("--val_ratio", type = float, default = 0.0 )
    
    parser.add_argument("--max_length", type = int, default = 100 )
    parser.add_argument("--total_spec_columns", type = int, default = 1000 )
    parser.add_argument("--batch_size", type = int, default = 4 )
    parser.add_argument("--learning_rate", type = float, default = 3e-6 )
    parser.add_argument("--lr_schedule", default = "linear" )
    parser.add_argument("--max_to_keep", type = int, default = -1 )
    parser.add_argument("--seed", type = int, default = 66100 )
    parser.add_argument("--weight_decay", type = float, default = 0.01 )
    parser.add_argument("--warmup_steps", type = int, default = 100 )
    parser.add_argument("--freeze_encoder", type = int, default = 0 )
    parser.add_argument("--dropout", type = float, default = 0.0 )
    parser.add_argument("--num_workers", type = int, default = 4 )
    
    parser.add_argument("--clear_cluster_codebook", type = int, help="set the pretrained model's cluster_codebook to empty dict. This is used when we train the segmenter on a complete new dataset. Set this to 0 if you just want to slighlt finetune the model with some additional data with the same cluster naming rule.", default = 1 )
    
    args = parser.parse_args()

    if args.use_wandb:
        wandb.init( project = args.project, name = args.run_name )
        wandb.define_metric("current_step")
        wandb.define_metric( "epoch", step_metric="current_step")
        wandb.define_metric( "train/loss", step_metric="current_step")
        wandb.define_metric( "train/learning_rate", step_metric="current_step")
        wandb.define_metric( "validate/score", step_metric="current_step")
        wandb.define_metric( "validate/segment_score", step_metric="current_step")
        wandb.define_metric( "validate/frame_score", step_metric="current_step")

    if args.seed is not None:
        np.random.seed(args.seed)  
        
    if args.val_ratio == 0.0:
        args.validate_every = None
        args.validate_per_epoch= None

    create_if_not_exists(args.model_folder)

    if args.gpu_list is None:
        args.gpu_list = np.arange(args.n_device).tolist()
        
    device = torch.device(  "cuda:%d"%( args.gpu_list[0] ) if torch.cuda.is_available() else "cpu" )

    model, tokenizer = load_model( args.initial_model_path, args.total_spec_columns, args.dropout)

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

    segmenter = WhisperSegmenterForEval( model = model, tokenizer = tokenizer )

    if args.clear_cluster_codebook:
        segmenter.update_cluster_codebook( {} )

    scaler = torch.cuda.amp.GradScaler()

    audio_path_list_train, label_path_list_train = get_audio_and_label_paths( args.train_dataset_folder ) 

    default_config = determine_default_config(audio_path_list_train, label_path_list_train, args.total_spec_columns)
    ## store the default segmentation config
    segmenter.model.config.default_segmentation_config = default_config
    segmenter.default_segmentation_config = default_config
    
    cluster_codebook = get_cluster_codebook( label_path_list_train, segmenter.cluster_codebook )
    segmenter.update_cluster_codebook( cluster_codebook )
    
    audio_list_train, label_list_train = load_data(audio_path_list_train, label_path_list_train, cluster_codebook = cluster_codebook, n_threads = 20, default_config = default_config )

    if args.val_ratio > 0:
        (audio_list_train, label_list_train), ( audio_list_val, label_list_val ) = train_val_split( audio_list_train, label_list_train, args.val_ratio )

    audio_list_train, label_list_train = slice_audios_and_labels( audio_list_train, label_list_train, args.total_spec_columns )

    training_dataset = VocalSegDataset( audio_list_train, label_list_train, tokenizer, args.max_length, 
                                         args.total_spec_columns, model.module.config.species_codebook  )

    training_dataloader = DataLoader( training_dataset, batch_size = args.batch_size , shuffle = True , 
                                             worker_init_fn = lambda x:[np.random.seed( epoch  + x ),  
                                                                    torch.manual_seed( epoch + x) ], 
                                             num_workers = args.num_workers, 
                                             drop_last= True, 
                                             pin_memory = False
                                           )
    if len(training_dataloader) == 0:
        training_dataloader = DataLoader( training_dataset, batch_size = args.batch_size , shuffle = True , 
                                             worker_init_fn = lambda x:[np.random.seed( epoch  + x ),  
                                                                    torch.manual_seed( epoch + x) ], 
                                             num_workers = args.num_workers, 
                                             drop_last= False,  ## set drop_last to False to fully utilize small dataset
                                             pin_memory = False
                                           )
    ## if the training dataset is really too small, then trigger the error
    if len(training_dataloader) == 0:
        print("Error: Too few examples (less than a batch) for training! Exit!")
        sys.exit(1)
    
    if args.max_num_iterations is not None and args.max_num_iterations > 0:            
        args.max_num_epochs = int(np.ceil( args.max_num_iterations / len( training_dataloader )  ))
    else:
        assert args.max_num_epochs is not None and args.max_num_epochs > 0
        args.max_num_iterations = len( training_dataloader ) * args.max_num_epochs
        
        if args.min_num_iterations is not None:
            args.max_num_iterations = max( args.max_num_iterations, args.min_num_iterations )
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
    val_score_history = []
    eary_stop = False
    current_step = 0

    progress = 0
    eta = None
    start_time = time.time()
    
    for epoch in range(args.max_num_epochs + 1):  # This +1 is to ensure current_step can reach args.max_num_iterations
        for count, batch in enumerate( tqdm( training_dataloader ) ):
            training_loss_value_list.append( train_iteration(batch) )
            
            if scheduler is not None:
                scheduler.step()
                
            current_step += 1

            current_time = time.time()
            current_progress = int(np.round(current_step / args.max_num_iterations * 100))
            eta = int((current_time - start_time) / ( current_step / args.max_num_iterations ) * ( 1 - current_step / args.max_num_iterations ))
            eta_hours = eta // 3600
            eta_minutes = (eta % 3600) // 60
            eta_seconds = ( eta % 3600 ) % 60
            if current_progress > progress:
                json.dump( { "progress":current_progress,
                             "eta": "%02d:%02d:%02d"%( eta_hours, eta_minutes, eta_seconds )
                           }, open( args.model_folder + "/status.json", "w" ) )
            progress = current_progress

            
            if current_step % args.print_every == 0:
                print("Epoch: %d, current_step: %d, learning rate: %f, Loss: %.4f"%( epoch, current_step, get_lr(optimizer)[0], np.mean(training_loss_value_list)) )
                if args.use_wandb:
                    wandb.log(
                        {
                            "current_step":current_step,
                            "train/learning_rate":get_lr(optimizer)[0],
                            "train/loss":np.mean(training_loss_value_list),
                            "epoch": epoch + count / len(training_dataloader)
                        }
                    )
                
                training_loss_value_list = [] 

            if ( args.validate_every is not None and current_step % args.validate_every == 0 ) or \
                ( args.validate_per_epoch and count == len(training_dataloader) - 1 ):
                print("Start validation ...")
                model.eval()
                ## in the validation set, set the num_trails to 1
                eval_res = evaluate( audio_list_val, label_list_val, segmenter, args.batch_size, args.max_length, num_trials =1, num_beams=1, target_cluster = None )
         
                print("Epoch: %d, current_step: %d, validation segment F1 score: %.2f, frame F1 score: %.2f"%( epoch, current_step, 
                                                                      eval_res["segment_wise"][-1], eval_res["frame_wise"][-1] ))
                if args.use_wandb:
                    wandb.log(
                        {
                            "current_step":current_step,
                            "validate/score": ( eval_res["segment_wise"][-1] + eval_res["frame_wise"][-1] ) * 0.5,
                            "validate/segment_score": eval_res["segment_wise"][-1],
                            "validate/frame_score": eval_res["frame_wise"][-1]
                        }
                    )    
                val_score_history.append( ( current_step, ( eval_res["segment_wise"][-1] + eval_res["frame_wise"][-1] ) * 0.5 ) )
                
                model.train()
            
            if ( args.save_every is not None and current_step % args.save_every == 0 ) or \
               ( args.save_per_epoch and count == len(training_dataloader) - 1 ):
                model.eval()
                save_model( model, tokenizer, current_step, args.model_folder, args.max_to_keep )
                model.train()

            if current_step >= 0.5 * args.max_num_iterations: ## training has been half-way done
                ## validation score keep decreasing for 2 validation steps
                if len( val_score_history ) >= 3 and \
                   val_score_history[-1][1] < val_score_history[-2][1] and \
                   val_score_history[-2][1] < val_score_history[-3][1]:
                    eary_stop = True
            
            if current_step >= args.max_num_iterations or eary_stop :
                if not os.path.exists( args.model_folder+"/checkpoint-%d"%(current_step) ):
                    model.eval()
                    save_model( model, tokenizer, current_step, args.model_folder, args.max_to_keep )
                break

        if current_step >= args.max_num_iterations or eary_stop :
            break   

    json.dump( { "progress":100,
                 "eta": "%02d:%02d:%02d"%( 0, 0, 0 )
               }, open( args.model_folder + "/status.json", "w" ) )

    best_checkpoint_batch_number = None
    if len(val_score_history) > 0:
        best_checkpoint_batch_number = sorted( val_score_history, key = lambda x:-x[1] )[0][0]
    else:
        ckpt_list = glob( args.model_folder + "/checkpoint-*" )
        if len( ckpt_list ) >0:
            ckpt_list.sort( key = os.path.getmtime )
            ckpt_name = ckpt_list[-1]
            best_checkpoint_batch_number = int(ckpt_name.split("-")[-1])
        

    if best_checkpoint_batch_number is not None:
        print("The best checkpoint on validation set is: %s," % ( args.model_folder+"/checkpoint-%d"%(best_checkpoint_batch_number) ) )
        os.system( "cp -r %s %s"%( args.model_folder+"/checkpoint-%d"%(best_checkpoint_batch_number), args.model_folder+"/final_checkpoint"  ) )
        ### remove other checkpoints
        os.system( "rm -r %s"%( args.model_folder+"/checkpoint-*" ) )

        hf_model_folder = args.model_folder+"/final_checkpoint"
        ct2_model_folder = hf_model_folder + "_ct2"
        
        subprocess.run([ "python", os.path.join( script_dirname, "convert_hf_to_ct2.py" ), 
                         "--model", hf_model_folder,
                         "--output_dir", ct2_model_folder,
                         "--quantization", "int8_float16"
                       ])
    try:
        os.remove( args.model_folder + "/status.json" )
    except:
        pass
    
    print("All Done!")    

