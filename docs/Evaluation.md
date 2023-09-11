# Evaluation

**Note**: 
* Runing the following commands in the main folder of the repository (where the README.md and .py files are located)
* For visualization of the spectrogram and model predictions, run the corresponding code in jupyter notebook.

## Evaluate the model using the frame-wise F1 score ($F1_\text{frame}$) and segment-wise F1 score ($F1_\text{seg}$)


```python
from model import WhisperSegmenterFast, WhisperSegmenter
import librosa
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from train import evaluate
from datautils import get_audio_and_label_paths
import os
from audio_utils import SpecViewer
import subprocess
from glob import glob
import json
```


```python
def evaluate_dataset( dataset_folder, model_path, num_trials, consolidation_method = "clustering",
                      max_length = 448, num_beams = 4, batch_size = 8 ):
    audio_list, label_list = [], []
    audio_paths, label_paths = get_audio_and_label_paths(dataset_folder)
    for audio_path, label_path in zip(audio_paths, label_paths):
        label = json.load( open( label_path ) )
        audio, _ = librosa.load( audio_path, sr = label["sr"] )
        audio_list.append(audio)
        label_list.append(label) 


    segmenter = WhisperSegmenterFast(  model_path = model_path,  device = "cuda")
    res = evaluate( audio_list, label_list, segmenter, batch_size, max_length, num_trials, consolidation_method, num_beams, target_cluster = None )

    all_res = {
        "segment_wise_scores": {"N-true-positive": res["segment_wise"][0],
                                "N-positive-in-prediction": res["segment_wise"][1],
                                "N-positive-in-ground-truth": res["segment_wise"][2],
                                "precision": res["segment_wise"][3],
                                "recall": res["segment_wise"][4],
                                "F1": res["segment_wise"][5]
                                },
        "frame_wise_scores": {"N-true-positive": res["frame_wise"][0],
                                "N-positive-in-prediction": res["frame_wise"][1],
                                "N-positive-in-ground-truth": res["frame_wise"][2],
                                "precision": res["frame_wise"][3],
                                "recall": res["frame_wise"][4],
                                "F1": res["frame_wise"][5]
                                }
    }
    return all_res
```

**Illustration of the parameters**
* dataset_folder:  The folde of a test dataset
* model_path: The path to a CTranslate model, e.g., "model/whisperseg-large-mouse/final_checkpoint_ct2" or our pretrained checkpoint "nccratliri/whisperseg-large-ms-ct2"
* num_trials: Whether to use multi-variant majority voting. Setting num_trials to 1 for no majority voting (recommended for human dataset), and setting num_trials to 3 for tri-variant majority voting (recommended for animal datasets).


```python
evaluate_dataset( "data/mouse/test/", "nccratliri/whisperseg-large-ms-ct2", num_trials =3 )
```


    Fetching 12 files:   0%|          | 0/12 [00:00<?, ?it/s]



    Downloading (…)68253/.gitattributes:   0%|          | 0.00/1.52k [00:00<?, ?B/s]



    Downloading (…)f41ca68253/README.md:   0%|          | 0.00/28.0 [00:00<?, ?B/s]



    Downloading (…)el/added_tokens.json:   0%|          | 0.00/22.1k [00:00<?, ?B/s]



    Downloading (…)1ca68253/config.json:   0%|          | 0.00/12.1k [00:00<?, ?B/s]



    Downloading (…)cial_tokens_map.json:   0%|          | 0.00/2.08k [00:00<?, ?B/s]



    Downloading (…)odel/normalizer.json:   0%|          | 0.00/52.7k [00:00<?, ?B/s]



    Downloading (…)/hf_model/merges.txt:   0%|          | 0.00/494k [00:00<?, ?B/s]



    Downloading (…)okenizer_config.json:   0%|          | 0.00/805 [00:00<?, ?B/s]



    Downloading (…)/hf_model/vocab.json:   0%|          | 0.00/1.04M [00:00<?, ?B/s]



    Downloading (…)hf_model/config.json:   0%|          | 0.00/2.71k [00:00<?, ?B/s]



    Downloading (…)8253/vocabulary.json:   0%|          | 0.00/1.07M [00:00<?, ?B/s]



    Downloading model.bin:   0%|          | 0.00/3.08G [00:00<?, ?B/s]


    100%|███████████████████████████████████████████████████████████████████| 3/3 [01:01<00:00, 20.34s/it]





    {'segment_wise_scores': {'N-true-positive': 213,
      'N-positive-in-prediction': 215,
      'N-positive-in-ground-truth': 229,
      'precision': 0.9906976744186047,
      'recall': 0.9301310043668122,
      'F1': 0.9594594594594595},
     'frame_wise_scores': {'N-true-positive': 10590,
      'N-positive-in-prediction': 10715,
      'N-positive-in-ground-truth': 10852,
      'precision': 0.9883341110592627,
      'recall': 0.9758569848875783,
      'F1': 0.9820559187647796}}



## Evaluate the model by visualization


```python
from model import WhisperSegmenterFast
from audio_utils import SpecViewer
segmenter = WhisperSegmenterFast( "nccratliri/whisperseg-large-ms-ct2", device="cuda" )
spec_viewer = SpecViewer()
```


```python
sr = 32000  
min_frequency = 0
spec_time_step = 0.0025
min_segment_length = 0.01
eps = 0.02
num_trials = 3

audio_file = "data/example_subset/Zebra_finch/test_juveniles/zebra_finch_R3428_40932.29996086_1_24_8_19_56.wav"
label_file = audio_file[:-4] + ".json"
audio, _ = librosa.load( audio_file, sr = sr )
label = json.load( open(label_file) )

prediction = segmenter.segment(  audio, sr = sr, min_frequency = min_frequency, spec_time_step = spec_time_step,
                       min_segment_length = min_segment_length, eps = eps,num_trials = num_trials )
spec_viewer.visualize( audio = audio, sr = sr, min_frequency= min_frequency, prediction = prediction, label=label, 
                       window_size=15, precision_bits=1 )
```
![vis](../assets/res_zebra_finch_juveniles.png)


For detailed visualization examples and parameters setting, please refer to [README.md#Voice-Activity-Detection-Demo](../README.md#Voice-Activity-Detection-Demo)
