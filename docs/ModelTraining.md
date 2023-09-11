# Model Training

**Note**: Runing the following commands in the main folder of the repository (where the README.md and .py files are located)

After the dataset has been downloaded or built. We can start training WhisperSeg

```bash
python train.py --initial_model_path openai/whisper-large --train_dataset_folder data/multi-species/train/  --model_folder model/whisperseg-large-ms --gpu_list 0
```

## Illustration of parameters
* --initial_model_path: The initial checkpoint that is used to initailize the WhisperSeg before training. The value can be:
    * **openai/whisper-large** : the pretrained ASR Whisper model, large version
    * **openai/whisper-medium**
    * **openai/whisper-small**
    * **openai/whisper-base**
    * **openai/whisper-tiny**
    * fine-tuned WhisperSeg model, e.g., **nccratliri/whisperseg-large-ms**
      
* --train_dataset_folder: The folder of the training dataset
* --model_folder: The folder to save the checkpoints, after training ends, the final checkpoint will be stored as "**final_checkpoint**" in the model folder. In the meantime, the CTranslate version will be stored as "**final_checkpoint_ct2**" in the model folder for faster inference.
* --gpu_list 0:  A list of gpu device ids that is used for training.
  
  (Troubleshoot: if the training hangs when using multiple GPUs, try adding "NCCL_P2P_DISABLE=1" when running the training script: "NCCL_P2P_DISABLE=1 python train.py xxxxx" )

## Training epochs and maximum training steps
With the setting above, the training will run **3** epochs. This default setting works for most of the cases. 

When the training set is very tiny, e.g, containing just 50 annotated segments in a single audio, 3 epochs may only account for a few iterations. In this case, we can explicitly set the maximum number of training iterations. The modified training command will be:
```bash
python train.py --initial_model_path openai/whisper-large --model_folder MODEL_FOLDER_HERE --train_dataset_folder TRAIN_DATASET_FOLDER --max_num_iterations 500 --gpu_list 0
```
Setting a maximum training iteration of 500 works for tiny dataset.

For more training parameters please refer to [train.py](../train.py#L100)

## GPU memory usage
* Finetuning the large-version of WhisperSeg requires 40 GB GPU.
* Finetuning the smaller version of WhisperSeg, such as WhisperSeg-base, a 10 GB GPU is sufficient.


