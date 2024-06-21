# Evaluation

**Note**: 
* Runing the following commands in the main folder of the repository (where the README.md and .py files are located)
* For visualization of the spectrogram and model predictions, run the corresponding code in jupyter notebook.

## Evaluate the model on a dataset using the frame-wise F1 score ($F1_\text{frame}$) and segment-wise F1 score ($F1_\text{seg}$)


```python
from evaluate import evaluate_dataset
```


```python
evaluate_dataset( "data/example_subset/Mouse/test/", "nccratliri/whisperseg-large-ms-ct2", num_trials =3 )
```

    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.13s/it]





    {'segment_wise_scores': {'N-true-positive': 124,
      'N-positive-in-prediction': 125,
      'N-positive-in-ground-truth': 133,
      'precision': 0.992,
      'recall': 0.9323308270676691,
      'F1': 0.9612403100775192},
     'frame_wise_scores': {'N-true-positive': 5042,
      'N-positive-in-prediction': 5105,
      'N-positive-in-ground-truth': 5207,
      'precision': 0.9876591576885406,
      'recall': 0.9683118878432879,
      'F1': 0.9778898370830101}}



## Evaluate the model by visualization


```python
from model import WhisperSegmenterFast
from audio_utils import SpecViewer
import librosa
import json
segmenter = WhisperSegmenterFast( "nccratliri/whisperseg-large-ms-ct2", device="cuda" )
spec_viewer = SpecViewer()
```

    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.



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