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
import wandb
import argparse
from audio_utils import SpecViewer, WhisperSegFeatureExtractor
from utils import *
from model import *
from datautils import *
import subprocess
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, set_seed

""" DDP: define the custom scaled_all_reduce function """
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def scaled_all_reduce(tensors):
    gpus = torch.distributed.get_world_size()
    if gpus == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / gpus)
    return tensors
""" DDP """

def train_iteration(batch, accumulation_steps):
    for key in batch:
        batch[key] = batch[key].to(device)
    
    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):   
        model_out = model( **batch )
        loss = model_out.loss.mean() / accumulation_steps
    scaler.scale(loss).backward()
    return loss.item()

def evaluate( audio_list, label_list, segmenter, batch_size, max_length, num_trials, consolidation_method = "clustering", num_beams=4, target_cluster = None ):

    total_n_true_positive_segment_wise, total_n_positive_in_prediction_segment_wise, total_n_positive_in_label_segment_wise = 0,0,0
    total_n_true_positive_frame_wise, total_n_positive_in_prediction_frame_wise, total_n_positive_in_label_frame_wise = 0,0,0
    
    for audio, label in tqdm(zip(audio_list, label_list), total = len(audio_list)):        
        prediction = segmenter.segment(  audio, sr = label["sr"],
                       min_frequency = label["min_frequency"],
                       spec_time_step = label["spec_time_step"],
                       min_segment_length = label["min_segment_length"],
                       eps = label["eps"],  ## for DBSCAN clustering
                       time_per_frame_for_voting = label.get("time_per_frame_for_voting", 0.001), ## for bin-wise voting, by default it is not used
                       consolidation_method = consolidation_method,
                       max_length = max_length, 
                       batch_size = batch_size, 
                       num_trials = num_trials,
                       num_beams = num_beams
                 )

        
        TP, P_pred, P_label = segmenter.segment_score( prediction, label,  target_cluster = target_cluster, tolerance = label["tolerance"] )[:3]
        total_n_true_positive_segment_wise += TP
        total_n_positive_in_prediction_segment_wise += P_pred
        total_n_positive_in_label_segment_wise += P_label
        
        
        TP, P_pred, P_label = segmenter.frame_score( prediction, label,  target_cluster = target_cluster, 
                                                     time_per_frame_for_scoring = label["time_per_frame_for_scoring"] )[:3]
        
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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--initial_model_path" )
    parser.add_argument("--model_folder" )
    parser.add_argument("--train_dataset_folder" )
    parser.add_argument("--project", default = "whisperseg-multi-species" )
    parser.add_argument("--run_name", default = None )
    parser.add_argument("--print_every", type = int, default = 100 )
    parser.add_argument("--validate_every", type = int, default = None )
    parser.add_argument("--validate_per_epoch", type = int, default = 0 )
    parser.add_argument("--save_every", type = int, default = None )
    parser.add_argument("--save_per_epoch", type = int, default = 0 )
    parser.add_argument("--max_num_epochs", type = int, default = 3 )
    parser.add_argument("--max_num_iterations", type = int, default = None )
    parser.add_argument("--val_ratio", type = float, default = 0.0 )
    parser.add_argument("--audio_mixing_ratio", type = float, default = 0.0 )
    
    parser.add_argument("--max_length", type = int, default = 100 )
    parser.add_argument("--total_spec_columns", type = int, default = 1000 )
    parser.add_argument("--batch_size_per_device", type = int, default = 4 )
    parser.add_argument("--learning_rate", type = float, default = 3e-6 )
    parser.add_argument("--lr_schedule", default = "cosine" )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before updating model parameters")
    parser.add_argument("--max_to_keep", type = int, default = -1 )
    parser.add_argument("--seed", type = int, default = 66102 )
    parser.add_argument("--weight_decay", type = float, default = 0.01 )
    parser.add_argument("--warmup_steps", type = int, default = 100 )
    parser.add_argument("--freeze_encoder", type = int, default = 0 )
    parser.add_argument("--dropout", type = float, default = 0.0 )
    parser.add_argument("--num_workers", type = int, default = 2 )
    parser.add_argument("--clear_cluster_codebook", type = int, help="set the pretrained model's cluster_codebook to empty dict. This is used when we train the segmenter on a complete new dataset. Set this to 0 if you just want to slighlt finetune the model with some additional data with the same cluster naming rule.", default = 0 )
    
    args = parser.parse_args()

    """ DDP: initially setting up the rank, world_size, and determine the ddp_mode """  
    try:
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        ddp_mode = True
    except:
        world_size = 1
        rank = 0
        local_rank = 0
        ddp_mode = False
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    """ DDP """

    ## set the main seed first to get a consistent train val split across different processes
    set_seed(args.seed) 

    ## All the printing, logging and checkpoint saving functions are only executed on the main process (rank=0)
    if rank == 0:
        wandb.init( project = args.project, name = args.run_name )
        wandb.define_metric("current_step")
        wandb.define_metric( "epoch", step_metric="current_step")
        wandb.define_metric( "train/loss", step_metric="current_step")
        wandb.define_metric( "train/learning_rate", step_metric="current_step")
        wandb.define_metric( "validate/score", step_metric="current_step")
        wandb.define_metric( "validate/segment_score", step_metric="current_step")
        wandb.define_metric( "validate/frame_score", step_metric="current_step")

        create_if_not_exists(args.model_folder)

    ## initializing the model, send model to device, applying DDP, and set up scaler and optimizer
    model, tokenizer = load_model( args.initial_model_path, args.total_spec_columns, args.dropout)
    model = model.to(device)
    if args.freeze_encoder:
        for para in model.model.encoder.parameters():
            para.requires_grad = False
    else:
        for para in model.model.encoder.parameters():
            para.requires_grad = True
    if ddp_mode:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)            
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate )
    scaler = torch.cuda.amp.GradScaler()
    segmenter = WhisperSegmenter( model = model, tokenizer = tokenizer )
    if args.clear_cluster_codebook:
        segmenter.update_cluster_codebook( {} )


    ## Loading the training dataset and evaluation dataset. 
    ### For the training set, load the whole dataset, and applying a DistributedSampler 
    ### For the evaluation set, only load the part that belongs to the current process, doing evaluation, and then merging the evaluation results by scaled_all_reduce
    audio_path_list_train, label_path_list_train = get_audio_and_label_paths( args.train_dataset_folder ) 
    audio_path_list_train, label_path_list_train = list(zip(*sorted(zip(audio_path_list_train, label_path_list_train ), key = lambda x:x[0] )))
    joint_audio_label_paths = list(zip( audio_path_list_train, label_path_list_train ))
    np.random.shuffle( joint_audio_label_paths )
    audio_path_list_train, label_path_list_train = list(zip(*joint_audio_label_paths))
    
    cluster_codebook = get_cluster_codebook( label_path_list_train, segmenter.cluster_codebook )
    segmenter.update_cluster_codebook( cluster_codebook )
    audio_list_train, label_list_train = load_data(audio_path_list_train, label_path_list_train, cluster_codebook = cluster_codebook, n_threads = 20 )

    if args.val_ratio > 0:
        (audio_list_train, label_list_train), ( audio_list_val, label_list_val ) = train_val_split( audio_list_train, label_list_train, args.val_ratio )
        #### only load the part that belongs to the current process
        val_shard_size = int( np.ceil(len(audio_list_val) / world_size )  )
        audio_list_val = audio_list_val[ rank*val_shard_size : (rank+1)*val_shard_size ]
        label_list_val = label_list_val[ rank*val_shard_size : (rank+1)*val_shard_size ]
    else:
        args.validate_every = None
        args.validate_per_epoch= None
    audio_list_train, label_list_train = slice_audios_and_labels( audio_list_train, label_list_train, args.total_spec_columns )

    ## After the training val random splitting is done and their is no need for consistent random seed across different processes, then setting a different seed for differnt process
    set_seed( args.seed + rank ) 

    species_codebook = model.module.config.species_codebook if ddp_mode else model.config.species_codebook
    training_dataset = VocalSegDataset( audio_list_train, label_list_train, tokenizer, args.max_length, 
                                         args.total_spec_columns, species_codebook, args.audio_mixing_ratio  )

    if ddp_mode:
        training_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, shuffle=True)
    else:
        training_sampler = None
    training_dataloader = DataLoader( training_dataset, 
                                      batch_size = args.batch_size_per_device, 
                                      shuffle = not ddp_mode, 
                                      sampler=training_sampler, 
                                      worker_init_fn = lambda x:set_seed( args.seed + rank + x ), 
                                      num_workers = args.num_workers , 
                                      drop_last= True,
                                      pin_memory = False
                                    )
    if len(training_dataloader) == 0:
        print("Error: Too few examples (less than a batch) for training! Exit!")
        sys.exit(1)

    ## Update the varaibles that are used to control the life cycle of the training process
    if args.max_num_iterations is not None and args.max_num_iterations > 0:
        args.max_num_epochs = int(np.ceil( args.max_num_iterations * args.gradient_accumulation_steps * args.batch_size_per_device * world_size / len( training_dataset )  ))
    else:
        assert args.max_num_epochs is not None and args.max_num_epochs > 0
        args.max_num_iterations = int(len( training_dataset )/( args.gradient_accumulation_steps * args.batch_size_per_device * world_size ) * args.max_num_epochs)
                
    if args.lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps= args.warmup_steps, 
            num_training_steps = args.max_num_iterations
        )
    elif args.lr_schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
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

    accumulation_counter = 0
    training_accumulation_loss = 0
    
    for epoch in range(args.max_num_epochs + 1):  # This +1 is to ensure current_step can reach args.max_num_iterations
        if rank == 0:
            pbar = tqdm(total=len(training_dataloader)//args.gradient_accumulation_steps  )
        for count, batch in enumerate(  training_dataloader  ):
            loss = train_iteration(batch, args.gradient_accumulation_steps)
            accumulation_counter += 1
            training_accumulation_loss += loss
            
            if accumulation_counter % args.gradient_accumulation_steps == 0 or count == len(training_dataloader) - 1 :
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                training_loss_value_list.append( training_accumulation_loss )
                
                accumulation_counter = 0
                training_accumulation_loss = 0
            
                current_step += 1
                if rank == 0:
                    pbar.update(1)

                if current_step % args.print_every == 0:
                    avg_loss = np.mean( training_loss_value_list )
                    """ DDP """
                    if ddp_mode:
                        avg_loss = scaled_all_reduce( [ torch.Tensor([ avg_loss ]).to(device) ] )[0].item()
                    """ DDP """
                    training_loss_value_list = []

                    if rank == 0:
                        print("Epoch: %d, current_step: %d, learning rate: %f, Loss: %.4f"%( epoch, current_step, get_lr(optimizer)[0], avg_loss) )
                        wandb.log(
                            {
                                "current_step":current_step,
                                "train/learning_rate":get_lr(optimizer)[0],
                                "train/loss":avg_loss,
                                "epoch": epoch + count / len(training_dataloader)
                            }
                        )

                if ( args.validate_every is not None and current_step % args.validate_every == 0 ) or \
                    ( args.validate_per_epoch and count == len(training_dataloader) - 1 ):
                    print("Start validation ...")
                    model.eval()
                    ## in the validation set, set the num_trails to 1
                    eval_res = evaluate( audio_list_val, label_list_val, segmenter, args.batch_size_per_device, args.max_length, num_trials =1, consolidation_method = None, num_beams=1, target_cluster = None )
                    eval_res = [  eval_res["segment_wise"][-1], eval_res["frame_wise"][-1] ]
                    model.train()

                    """ DDP """
                    if ddp_mode:
                        eval_res = scaled_all_reduce( [torch.Tensor(eval_res).to(device)] )[0].detach().cpu().numpy().tolist()
                    """ DDP """
                    val_score_history.append( ( current_step, ( eval_res[0] + eval_res[1] ) * 0.5 ) )
                    
                    if rank == 0:
                        print("Epoch: %d, current_step: %d, validation segment F1 score: %.2f, frame F1 score: %.2f"%( epoch, current_step, 
                                                                        eval_res[0], eval_res[1] ))
                        wandb.log(
                            {
                                "current_step":current_step,
                                "validate/score": ( eval_res[0] + eval_res[1] ) * 0.5,
                                "validate/segment_score": eval_res[0],
                                "validate/frame_score": eval_res[1]
                            }
                        )    
                    
                if ( args.save_every is not None and current_step % args.save_every == 0 ) or \
                ( args.save_per_epoch and count == len(training_dataloader) - 1 ):
                    if rank == 0:
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
                        if rank == 0:
                            model.eval()
                            save_model( model, tokenizer, current_step, args.model_folder, args.max_to_keep )
                    break

        if current_step >= args.max_num_iterations or eary_stop :
            break   

    ## Get the best checkpoint and save it to disk
    if rank == 0:
        best_checkpoint_batch_number = None
        if len(val_score_history) > 0:
            best_checkpoint_batch_number = sorted( val_score_history, key = lambda x:-x[1] )[0][0]
        else:
            ckpt_list = glob( args.model_folder + "/*" )
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
            
            subprocess.run([ "python", "convert_hf_to_ct2.py", 
                            "--model", hf_model_folder,
                            "--output_dir", ct2_model_folder,
                            "--quantization", "float16"
                        ])    
        
        print("All Done!")   

