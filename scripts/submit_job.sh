#!/bin/bash
  
### Here are the SBATCH parameters that you should always consider:
#SBATCH --time=1-00:00:00    ## days-hours:minutes:seconds
#SBATCH --mem 128G         ## 3GB ram (hardware ratio is < 4GB/core)
#SBATCH --ntasks=1          ## Not strictly necessary because default is 1
#SBATCH --cpus-per-task=16   ## Use greater than 1 for parallelized jobs
#SBATCH --gres=gpu:A100:1


module load anaconda3

source activate wseg

cd ..

# python train.py --initial_model_path openai/whisper-large --train_dataset_folder data/datasets/zebra_finch/train/ --model_folder model/whisperseg-zebra-finch-vad --gpu_list 0

# python train.py --initial_model_path openai/whisper-large --train_dataset_folder data/datasets/animals/train/ --model_folder model/whisperseg-animal-vad --gpu_list 0

python train.py --initial_model_path model/whisperseg-animal-vad/final_checkpoint --train_dataset_folder data/datasets/meerkat/data/train/ --model_folder model/whisperseg-meerkat-animal-based-v2.0 --gpu_list 0 --max_num_iterations 5000

python train.py --initial_model_path model/whisperseg-zebra-finch-vad/final_checkpoint --train_dataset_folder data/datasets/meerkat/data/train/ --model_folder model/whisperseg-meerkat-zebra-finch-based-v2.0 --gpu_list 0 --max_num_iterations 5000


