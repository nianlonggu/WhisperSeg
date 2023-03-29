# Set up environment

## Option 1: Set up environment in your local computer
1. Install Miniconda (or Anaconda)
https://docs.conda.io/en/latest/miniconda.html

2. In the "base" anaconda environment, create a new environment "syllable_segment" (This installation will take a few minutes.):
```bash
conda env create -f environment.yml
```
3. Activate the anaconda environment "syllable_segment":
```bash
conda activate syllable_segment
```
Alternatively, one can go through all the packages that are imported, and install the missing packages manually.

## Option 2: Run the code on google colab (with GPU runtime)
Since google colab has preinstalled most of the package, we only need to install the "transformers" package
```bash
pip install transformers
```

The following commands assume that this jupyter notebook is running within the created anaconda environment, or with all necessary python packages installed.

# Use the pretrained WhisperSeg in command line


```python
from model import Segmenter
import librosa
import pandas as pd
import numpy as np
```


```python
# initialize the segmenter
segmenter = Segmenter(  model_path = "nianlong/vocal-segment-zebra-finch-whisper-large", 
                        device = "cuda")
```


```python
# load an audio file, and resample the audio to the sampling rate 16000 Hz 
audio_file_name = "data/R3406_035/test/R3406_40911.54676404_1_3_15_11_16.wav"
audio, _ = librosa.load( audio_file_name, sr = 16000 )
```


```python
# segment the audio, i.e., predict the paired on/offset of the audio
prediction = segmenter.segment( audio )
```


```python
prediction
```




    {'onset': array([1.47, 1.83, 1.93, 2.06, 2.17, 2.25, 2.42, 2.71, 2.91, 3.05, 3.13,
            3.28, 3.62, 3.81, 4.09, 4.2 , 4.97]),
     'offset': array([1.57, 1.88, 2.03, 2.1 , 2.24, 2.37, 2.58, 2.84, 3.02, 3.12, 3.24,
            3.43, 3.67, 3.9 , 4.15, 4.38, 5.07])}



To save the prediction into a .csv file, run the following command:


```python
# pd.DataFrame(prediction).to_csv("predicted_annotations.csv", index = False)
```

# Visualize the prediction
This visualize function is only supported on jupter notebook and google colab, because it is an interactive plot that is a ipywidget feature.


```python
segmenter.visualize( audio = audio, prediction = prediction, audio_file_name = audio_file_name)
```


    interactive(children=(FloatSlider(value=1.0, description='offset', max=2.0530625000000002), Output()), _dom_cl…





    <function ipywidgets.widgets.interaction._InteractFactory.__call__.<locals>.<lambda>(*args, **kwargs)>



If we know the ground-truth label, we can also plot both the predicted label and the ground-truth label to visualize the prediction error.

For example, we have the annotation file for the following wav file:


```python
human_annotation_file_name = audio_file_name[:-4]+".csv"
audio_file_name, human_annotation_file_name
```




    ('data/R3406_035/test/R3406_40911.54676404_1_3_15_11_16.wav',
     'data/R3406_035/test/R3406_40911.54676404_1_3_15_11_16.csv')




```python
""" 
Both label and prediction is a dictionary. 
The dictionary contains two keys: onset and offset. 
The value for each key is an numpy array
"""
label_df = pd.read_csv( human_annotation_file_name )
label = {
    "onset":np.array(label_df["onset"]),
    "offset":np.array(label_df["offset"])
}
```


```python
segmenter.visualize( audio = audio, prediction = prediction, label = label, audio_file_name = audio_file_name)
```


    interactive(children=(FloatSlider(value=1.0, description='offset', max=2.0530625000000002), Output()), _dom_cl…





    <function ipywidgets.widgets.interaction._InteractFactory.__call__.<locals>.<lambda>(*args, **kwargs)>



# Finetune WhisperSeg

We are going to finetune WhisperSeg on the zebra finch dataset released by the DAS paper. In this dataset the researchers have adopted a very different standard when segmenting the syllables. So the model "vocal-segment-zebra-finch-whisper-large" that was pretrained on Tomas's dataset will perform not well. Let's have a look:


```python
audio_file_name = "data/DAS_zebra_finch/test/birdname_130519_113316.31.wav"
human_annotation_file_name = "data/DAS_zebra_finch/test/birdname_130519_113316.31.csv"
audio, _ = librosa.load( audio_file_name, sr = 16000 )
label_df = pd.read_csv( human_annotation_file_name )
label = {
    "onset":np.array(label_df["onset"]),
    "offset":np.array(label_df["offset"])
}
prediction = segmenter.segment( audio )
segmenter.visualize( audio = audio, prediction = prediction, label = label, audio_file_name = audio_file_name)
```


    interactive(children=(FloatSlider(value=4.3, description='offset', max=8.6498125), Output()), _dom_classes=('w…





    <function ipywidgets.widgets.interaction._InteractFactory.__call__.<locals>.<lambda>(*args, **kwargs)>



There are quite a lot of False Positives! That's why we need to finetune WhisperSeg.

## Dataset preparation
Before finetuning WhisperSeg, we need to first prepare the training dataset and the test dataset.

Take the training dataset as an example: 
* All training audio and annotation files should be placed in the same folder.
* The audio file should have a format ".wav" (lowercase),  and the annotation has a format ".csv"
* The names of the audio file and the corresponding annotation file should be matched. For example, if there is an audio file named "XXXXX_bird_12345.wav", the corresponding annotation file needs to be named as "XXXXX_bird_12345.csv"
* Inside the annotation file, there will be two columns: "onset" and "offset". The unit of the value is second.
* The .wav file can have various sampling rate. The model will resample them to 16kHz automatically.

For the testing dataset, the requirement is the same.

Please check the folder: data/DAS_zebra_finch/ for concrete examples.

## Training


Explanation of the training parameters:
* initial_model_path: The initail checkpoint of Wshiper, here we use the whisper model pretrained on Tomas's dataset
* model_folder: the folder to save the trained checkpoint
* result_folder: the folder to save some loging information and validation and test results
* train_dataset_folder: the folder that contains all the paired training audio and annotation data as described above
* test_dataset_folder: 
* warmup_steps: the learning rate will increase from 0 linearly to 1e-6 within warmup steps
* save_every: save the checkpoint after save_every training steps
* max_num_iterations: the maximum number of training steps before the training finishes.
* batch_size: Training WhisperSeg requires around 40 GB GPU RAM if we use a batch size of 4. For smaller GPU, please try batch size 2 or 1.

Since the DAS_zebra_finch is a very small dataset, we finetune WhisperSeg for 1000 steps, and set up the warmup step to 200. 
We do not create validation set, and use all the training set to train the model until the max_num_iterations is reached, and we only keep the model checkpont at the max_num_iterations. Empirically this works very stable.


```python
!python train.py -initial_model_path nianlong/vocal-segment-zebra-finch-whisper-large -model_folder model/DAS_zebra_finch -result_folder result/DAS_zebra_finch -train_dataset_folder data/DAS_zebra_finch/train -test_dataset_folder data/DAS_zebra_finch/test -warmup_steps 200 -save_every 1000 -max_num_iterations 1000 -batch_size 4

```

    /home/nianlong/miniconda3/envs/syllable_segment/lib/python3.9/site-packages/transformers/optimization.py:306: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(
      0%|                                                    | 0/13 [00:00<?, ?it/s]/home/nianlong/miniconda3/envs/syllable_segment/lib/python3.9/site-packages/torch/optim/lr_scheduler.py:138: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
      warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    100%|███████████████████████████████████████████| 13/13 [00:10<00:00,  1.28it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.41it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.38it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.41it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
     62%|███████████████████████████                 | 8/13 [00:05<00:03,  1.41it/s]Epoch: 7, current_batch: 100, learning rate: 0.000000, Loss: 1.1831
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.36it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.36it/s]
    100%|███████████████████████████████████████████| 13/13 [00:10<00:00,  1.29it/s]
     31%|█████████████▌                              | 4/13 [00:03<00:06,  1.34it/s]Epoch: 15, current_batch: 200, learning rate: 0.000001, Loss: 0.3107
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.36it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.36it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.36it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.36it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.36it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.36it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.35it/s]
      0%|                                                    | 0/13 [00:00<?, ?it/s]Epoch: 23, current_batch: 300, learning rate: 0.000001, Loss: 0.1981
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.35it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.36it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.36it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.36it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.36it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
     69%|██████████████████████████████▍             | 9/13 [00:06<00:02,  1.41it/s]Epoch: 30, current_batch: 400, learning rate: 0.000001, Loss: 0.1623
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
     38%|████████████████▉                           | 5/13 [00:03<00:05,  1.37it/s]Epoch: 38, current_batch: 500, learning rate: 0.000001, Loss: 0.1604
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.38it/s]
      8%|███▍                                        | 1/13 [00:01<00:12,  1.02s/it]Epoch: 46, current_batch: 600, learning rate: 0.000000, Loss: 0.1527
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
     77%|█████████████████████████████████          | 10/13 [00:07<00:02,  1.43it/s]Epoch: 53, current_batch: 700, learning rate: 0.000000, Loss: 0.1412
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.38it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.38it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
     46%|████████████████████▎                       | 6/13 [00:04<00:05,  1.39it/s]Epoch: 61, current_batch: 800, learning rate: 0.000000, Loss: 0.1327
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.36it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.36it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.36it/s]
     15%|██████▊                                     | 2/13 [00:01<00:09,  1.20it/s]Epoch: 69, current_batch: 900, learning rate: 0.000000, Loss: 0.1299
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.38it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.38it/s]
    100%|███████████████████████████████████████████| 13/13 [00:09<00:00,  1.37it/s]
     85%|████████████████████████████████████▍      | 11/13 [00:07<00:01,  1.43it/s]Epoch: 76, current_batch: 1000, learning rate: 0.000000, Loss: 0.1284
     85%|████████████████████████████████████▍      | 11/13 [00:14<00:02,  1.33s/it]
    The best checkpoint on validation set is: model/DAS_zebra_finch/checkpoint-1000,
    Reporting test results ...
    100%|█████████████████████████████████████████████| 2/2 [00:10<00:00,  5.47s/it]
    Test performance: f1 score: 0.9720
    Removing sub-optimal checkpoints ...
    All Done!


Let's use the finetuned WhisperSeg to segment the audio from the DAS_zebra_finch test set again.


```python
from model import Segmenter
import librosa
import pandas as pd
import numpy as np

segmenter = Segmenter(  model_path = "model/DAS_zebra_finch/checkpoint-1000", 
                        device = "cuda")

audio_file_name = "data/DAS_zebra_finch/test/birdname_130519_113316.31.wav"
human_annotation_file_name = "data/DAS_zebra_finch/test/birdname_130519_113316.31.csv"
audio, _ = librosa.load( audio_file_name, sr = 16000 )
label_df = pd.read_csv( human_annotation_file_name )
label = {
    "onset":np.array(label_df["onset"]),
    "offset":np.array(label_df["offset"])
}
prediction = segmenter.segment( audio )
segmenter.visualize( audio = audio, prediction = prediction, label = label, audio_file_name = audio_file_name)
```


    interactive(children=(FloatSlider(value=4.3, description='offset', max=8.6498125), Output()), _dom_classes=('w…





    <function ipywidgets.widgets.interaction._InteractFactory.__call__.<locals>.<lambda>(*args, **kwargs)>



Therefore, WhisperSeg does perform better after finetuning!

# Speed Up Inference with ctranslate2 - SegmenterFast

The environment.yml has been updated due to the adding of the ctranslate2 package. 

Running the code below does not rely on the previous codes in this notebook. One can restart the kernel before run the following code, to release some GPU usage.

## convert the huggingface Whisper model to the CTranslate2 model, and store the configuration files of the tokenizer and feature-extractor

Note: The following cell only needs to be run once.


```python
from huggingface_hub import hf_hub_download
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
import os
## If you have trained model on new dataset, replace this hf_model_path's value with the path to the newly saved checkpoint
hf_model_path = "nianlong/vocal-segment-zebra-finch-whisper-large"
## The path to the folder where the converted ctranslate2 model will be saved. 
## In the meantime, the configuration files for Tokenizer and FeatureExtractors will also be copied to this folder
ct2_model_path = "model/vocal-segment-zebra-finch-whisper-large-ct2"

assert not os.path.exists(ct2_model_path)

os.system( "ct2-transformers-converter --model %s --output_dir %s"%( hf_model_path, ct2_model_path ) )
## copy the configuration file of the original huggingface model, because it contains some useful hyperparameters
hf_hub_download(repo_id=hf_model_path, filename="config.json", local_dir = ct2_model_path+"/hf_model/")
WhisperFeatureExtractor.from_pretrained( hf_model_path ).save_pretrained( ct2_model_path+"/hf_model/" )
WhisperTokenizer.from_pretrained(hf_model_path, language = "english" ).save_pretrained( ct2_model_path+"/hf_model/" )
```




    ('model/vocal-segment-zebra-finch-whisper-large-ct2/hf_model/tokenizer_config.json',
     'model/vocal-segment-zebra-finch-whisper-large-ct2/hf_model/special_tokens_map.json',
     'model/vocal-segment-zebra-finch-whisper-large-ct2/hf_model/vocab.json',
     'model/vocal-segment-zebra-finch-whisper-large-ct2/hf_model/merges.txt',
     'model/vocal-segment-zebra-finch-whisper-large-ct2/hf_model/normalizer.json',
     'model/vocal-segment-zebra-finch-whisper-large-ct2/hf_model/added_tokens.json')



## Use the CTranslate2 Converted Model


```python
from model import SegmenterFast
import librosa
import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm
```


```python
segmenter_fast = SegmenterFast( "model/vocal-segment-zebra-finch-whisper-large-ct2", device="cuda" )
```


```python
audio_file_name = "data/R3406_035/test/R3406_40911.54676404_1_3_15_11_16.wav"
human_annotation_file_name = "data/R3406_035/test/R3406_40911.54676404_1_3_15_11_16.csv"
audio, _ = librosa.load( audio_file_name, sr = 16000 )
label_df = pd.read_csv( human_annotation_file_name )
label = {
    "onset":np.array(label_df["onset"]),
    "offset":np.array(label_df["offset"])
}

tic = time.time()
prediction = segmenter_fast.segment( audio )
tac = time.time()

segmenter_fast.visualize( audio = audio, prediction = prediction, label = label, audio_file_name = audio_file_name)

print("Audio Length: %f s"%(len(audio)/16000))
print("Segmentation Time: %f s"%(tac - tic))
```


    interactive(children=(FloatSlider(value=1.0, description='offset', max=2.0530625000000002), Output()), _dom_cl…


    Audio Length: 7.053063 s
    Segmentation Time: 1.869219 s



```python
audio_file_name = "data/R3277/R3277_40905.13765404_12_28_3_49_25.wav"
audio, _ = librosa.load( audio_file_name, sr = 16000 )

tic = time.time()
prediction = segmenter_fast.segment( audio )
tac = time.time()

segmenter_fast.visualize( audio = audio, prediction = prediction, audio_file_name = audio_file_name)

print("Audio Length: %f s"%(len(audio)/16000))
print("Segmentation Time: %f s"%(tac - tic))
```


    interactive(children=(FloatSlider(value=0.6000000000000001, description='offset', max=1.3739375000000003), Out…


    Audio Length: 6.373938 s
    Segmentation Time: 0.245195 s



```python
audio_file_name = "data/R3277/R3277_40905.38807_12_28_10_46_47.wav"
audio, _ = librosa.load( audio_file_name, sr = 16000 )

tic = time.time()
prediction = segmenter_fast.segment( audio )
tac = time.time()

segmenter_fast.visualize( audio = audio, prediction = prediction, audio_file_name = audio_file_name)

print("Audio Length: %f s"%(len(audio)/16000))
print("Segmentation Time: %f s"%(tac - tic))
```


    interactive(children=(FloatSlider(value=13.600000000000001, description='offset', max=27.2641875), Output()), …


    Audio Length: 32.264187 s
    Segmentation Time: 1.722064 s



```python
audio_file_name = "data/R3277/R3277_40905.406363_12_28_11_17_16.wav"
audio, _ = librosa.load( audio_file_name, sr = 16000 )

tic = time.time()
prediction = segmenter_fast.segment( audio )
tac = time.time()

segmenter_fast.visualize( audio = audio, prediction = prediction, audio_file_name = audio_file_name)

print("Audio Length: %f s"%(len(audio)/16000))
print("Segmentation Time: %f s"%(tac - tic))
```


    interactive(children=(FloatSlider(value=0.2, description='offset', max=0.4276875000000002), Output()), _dom_cl…


    Audio Length: 5.427688 s
    Segmentation Time: 0.357237 s


## Test the Speed of SegmenterFast

Here we use the SegmenterFast to segment all audios in the folder "data/DAS_zebra_finch/train/", and record the time spent.


```python
audio_list = [ librosa.load("data/DAS_zebra_finch/train/"+fname, sr = 16000)[0] for fname in os.listdir("data/DAS_zebra_finch/train/") if fname.endswith(".wav") ]

num_of_audio_files = len(audio_list)
total_audio_length = sum([ len(audio)/16000  for audio in audio_list ])

print("Total number of audio (.wav) files:", num_of_audio_files)
print("Total length of audio files: %f s"%(total_audio_length))
```

    Total number of audio (.wav) files: 14
    Total length of audio files: 175.500438 s



```python
tic = time.time()

for audio in tqdm(audio_list):
    segmenter_fast.segment(audio)
    
tac = time.time()
print("Total segmentation time: %f s for segmenting %.2f minutes audio"%(tac - tic, total_audio_length/60))
print("Average segmentation speed: %f s of audio segmented per second"%( total_audio_length / (tac - tic) ))
```

    100%|████████████████████████████████████████████████████████| 14/14 [00:19<00:00,  1.40s/it]

    Total segmentation time: 19.593630 s for segmenting 2.93 minutes audio
    Average segmentation speed: 8.957015 s of audio segmented per second


    


Note that the default num_trials is 3, which means one audio file will be segmented three times, each time with a slightly different offset. This helps to improve the segmentation accuracy, but it will slow down the segmentation process.

If num_trials is set to 1, the segmentation speed will be improved.


```python
tic = time.time()

for audio in tqdm(audio_list):
    segmenter_fast.segment(audio, num_trials= 1)
    
tac = time.time()
print("Total segmentation time: %f s for segmenting %.2f minutes audio"%(tac - tic, total_audio_length/60))
print("Average segmentation speed: %f s of audio segmented per second"%( total_audio_length / (tac - tic) ))
```

    100%|████████████████████████████████████████████████████████| 14/14 [00:11<00:00,  1.23it/s]

    Total segmentation time: 11.357352 s for segmenting 2.93 minutes audio
    Average segmentation speed: 15.452585 s of audio segmented per second


    


As a comparison, the speed of faster-whisper is 54s for transcribing 13 minutes of audio: https://github.com/guillaumekln/faster-whisper#large-v2-model-on-gpu 

## GPU Usage of SegmenterFast

GPU usage when idle: 3.8 GB <br>
GPU usage when segmenting (with a internal batch size 16):  up to 6 GB


```python

```
