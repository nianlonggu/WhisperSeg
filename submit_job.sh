#!/bin/bash
  
### Here are the SBATCH parameters that you should always consider:
#SBATCH --time=1-00:00:00    ## days-hours:minutes:seconds
#SBATCH --mem 256G         ## 3GB ram (hardware ratio is < 4GB/core)
#SBATCH --ntasks=1          ## Not strictly necessary because default is 1
#SBATCH --cpus-per-task=16   ## Use greater than 1 for parallelized jobs
#SBATCH --gres=gpu:A100:4


module load anaconda3

source activate wseg

torchrun --nproc_per_node=4 train.py --initial_model_path openai/whisper-large --model_folder model/whisperseg-large-vad-v3.0/ --train_dataset_folder data/datasets/multi_species_data/data/multi-species/train/ --gradient_accumulation_steps 1 --batch_size_per_device 16  --learning_rate 5e-6 --audio_mixing_ratio 0.5 --max_num_iterations 20000 --save_every 5000 --val_ratio 0.1 --validate_every 5000
