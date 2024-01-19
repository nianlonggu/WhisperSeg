#!/bin/bash
  
### Here are the SBATCH parameters that you should always consider:
#SBATCH --time=1-00:00:00    ## days-hours:minutes:seconds
#SBATCH --mem 256G         ## 3GB ram (hardware ratio is < 4GB/core)
#SBATCH --ntasks=1          ## Not strictly necessary because default is 1
#SBATCH --cpus-per-task=16   ## Use greater than 1 for parallelized jobs
#SBATCH --gres=gpu:A100:4


module load anaconda3

source activate wseg

NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 train.py --initial_model_path openai/whisper-large --model_folder model/whisperseg-large-vad/ --train_dataset_folder data/multi_species_data/data/multi-species/train/ --gradient_accumulation_steps 1 --batch_size_per_device 4  --max_num_epochs 5 --learning_rate 1e-5
