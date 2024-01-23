# Dataset Processing

**Note**: Runing the following commands in the main folder of the repository (where the README.md and .py files are located)

## Download the dataset used in the paper


```python
from huggingface_hub import snapshot_download
```

### Multi-Species VAD dataset
This dataset contains the union of the VAD datasets for five different species (zebra finch, bengalese finch, marmoset, mouse and human) used in the paper.


```python
snapshot_download('nccratliri/vad-multi-species', local_dir = "data/multi-species", repo_type="dataset" )
```

### Zebra finch
For the zebra finch only dataset, the training set contains the training examples for both adults and juveniles. For testing, we divide the test set into adults and juveniles sets and report the test performance separately.


```python
snapshot_download('nccratliri/vad-zebra-finch', local_dir = "data/zebra-finch", repo_type="dataset" )
```

### Bengalese finch


```python
snapshot_download('nccratliri/vad-bengalese-finch', local_dir = "data/bengalese-finch", repo_type="dataset" )
```

### Marmoset


```python
snapshot_download('nccratliri/vad-marmoset', local_dir = "data/marmoset", repo_type="dataset" )
```

### Mouse


```python
snapshot_download('nccratliri/vad-mouse', local_dir = "data/mouse", repo_type="dataset" )
```

### Human-AVA-Speech


```python
snapshot_download('nccratliri/vad-human-ava-speech', local_dir = "data/human-ava-speech", repo_type="dataset" )
```

## Build custom dataset

One dataset contains two subsets: training subset and testing subset, each stored in a separate folder, e.g., "train/" and "test/".

In the train/ folder, each audio recording is paired with an annotation file. Here are the requirements for the audio recording and the corresponding annotation file:
* audio recording:
  * It is a ".wav" file, e.g., the file name can be arbitrary but must end with ".wav, such as "rec_0001.wav".
  * It has only one channel (mono sound).
  * Its sampling rate and length (duration) can be arbitrary.
* annotation file:
  * It's a .json file, and its name should be the same as the corresponding audio file except for the format. So given an audio file named "rec_0001.wav", the corresponding annotation file should be "rec_0001.json"
  * The json file contains the following keys:
    * "**onset**": a list of the starting time (in second) of the segments in the audio, ordered chronologically.
    * "**offset**": a list of the ending time (in second) of segments in the audio
    * "**cluster**": a list of the segment types (plain text) of the segments in the audio. If in the dataset, there are multiple species, the segment type of different species should be unique. A recommended way is to prefix the species name before the segment type. For example "mouse_call_type_0", "marmoset_call_type_0", etc.
    * "**species**": The species in the audio, e.g., "zebra_finch". In this paper, we experinmented on five species: "zebra_finch", "bengalese_finch", "mouse", "marmoset", "human". Adding new species is possible. **When adding new species, go to the** [load_model() function in model.py](../model.py#L90), **add a new pair of species_name:species_token to the species_codebook variable. E.g., "meerkat":"<|meerkat|>"**.
    * "**sr**": The sampling rate that is used to load the audio. The audio file will be resampled to the sampling rate specified by **sr**, regardless of the native sampling rate of the audio file.
    * "**min_frequency**": the minimum frequency when computing the Log Melspectrogram. Frequency components below min_frequency will not be included in the input spectrogram.
    * "**spec_time_step**": Spectrogram Time Resolution. By default, one single input spectrogram of WhisperSeg contains 1000 columns. 'spec_time_step' represents the time difference between two adjacent columns in the spectrogram. It is equal to FFT_hop_size / sampling_rate: $\frac{L_\text{hop}}{f_s}$ .
    * "**min_segment_length**": The minimum allowed length of predicted segments. The predicted segments whose length is below 'min_segment_length' will be discarded.
    * "**tolerance**": When computing the $F1_\text{seg}$ score, we need to check if the both the absolute difference between the predicted onset and the ground-truth onset and the absolute difference between the predicted and ground-truth offsets are below a tolerance (in second). We choose **tolerance** 0.2 s for human and 0.01s for animals.
    * "**time_per_frame_for_scoring**": The time bin size (in second) used when computing the $F1_\text{frame}$ score. We set **time_per_frame_for_scoring** to 0.001 for all datasets.
    * "**eps**": The threshold $\epsilon_\text{vote}$ during the multi-trial majority voting when processing long audio files

*Recommended values of **sr, min_frequency, spec_time_step, min_segment_length, time_per_frame_for_scoring, and eps** are available at [config/segment_config.json](../config/segment_config.json)

The test/ folder contains the test set and has the same structure as the training set.

Here is the file structures (taking the marmoset dataset (downloaded above) as an example):
```
data/marmoset/
├── test
│   ├── marmoset_pair1_animal1_animal1out_0.json
│   ├── marmoset_pair1_animal1_animal1out_0.wav
│   ├── marmoset_pair2_animal1_animal1out_0.json
│   ├── marmoset_pair2_animal1_animal1out_0.wav
│   ├── marmoset_pair3_animal1_animal1out_0.json
│   ├── marmoset_pair3_animal1_animal1out_0.wav
│   ├── marmoset_pair3_animal1_together_0.json
│   ├── marmoset_pair3_animal1_together_0.wav
│   ├── marmoset_pair4_animal1_together_A_0.json
│   ├── marmoset_pair4_animal1_together_A_0.wav
│   ├── marmoset_pair4_animal1_together_B_0.json
│   ├── marmoset_pair4_animal1_together_B_0.wav
│   ├── marmoset_pair5_animal1_animal1out_0.json
│   └── marmoset_pair5_animal1_animal1out_0.wav
└── train
    ├── marmoset_pair1_animal1_animal1out_0.json
    ├── marmoset_pair1_animal1_animal1out_0.wav
    ├── marmoset_pair2_animal1_animal1out_0.json
    ├── marmoset_pair2_animal1_animal1out_0.wav
    ├── marmoset_pair3_animal1_animal1out_0.json
    ├── marmoset_pair3_animal1_animal1out_0.wav
    ├── marmoset_pair3_animal1_together_0.json
    ├── marmoset_pair3_animal1_together_0.wav
    ├── marmoset_pair4_animal1_together_A_0.json
    ├── marmoset_pair4_animal1_together_A_0.wav
    ├── marmoset_pair4_animal1_together_B_0.json
    ├── marmoset_pair4_animal1_together_B_0.wav
    ├── marmoset_pair5_animal1_animal1out_0.json
    └── marmoset_pair5_animal1_animal1out_0.wav
```

Here is how it looks like in an annotation file (take "marmoset_pair4_animal1_together_B_0.json" as an example):
```
{'onset': [0.1979075547210413,
  10.623481169473052,
  15.8850552318886,
  24.79427063612889,
  28.810797420332847,
  38.2856537289058,
  48.63584878094048,
  58.04026121482411,
  64.63873157548687,
  64.7831555952348,
  68.39671202322779,
  78.3215963108371,
  88.56905355060303,
  98.87149718874277,
  100.79554794271394,
  111.51698102894191,
  115.66329432038287,
  125.8880986515378,
  126.01743089368824,
  136.24306110239945,
  146.5839366300272,
  156.9141451247167],
 'offset': [0.6325124716552182,
  10.732488727627924,
  16.165301587301883,
  24.99787319137772,
  29.080698832384087,
  38.65104624298783,
  48.960403628118,
  58.40517025116287,
  64.71413591707028,
  64.94158551347277,
  68.73825700016414,
  78.5864520979826,
  88.86396659303114,
  99.44553737426918,
  101.51028706999728,
  111.81912465550818,
  115.90700986006686,
  125.97332390185488,
  126.26139656595478,
  136.58120794909837,
  146.90520472896583,
  157.0802205816326],
 'cluster': ['marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr',
  'marmoset_tr'],
 'species': 'marmoset',
 'sr': 48000,
 'min_frequency': 0,
 'spec_time_step': 0.0025,
 'min_segment_length': 0.01,
 'tolerance': 0.01,
 'time_per_frame_for_scoring': 0.001,
 'eps': 0.02}
```


The choice of the parameters are described in [README/Illustration-of-segmentation-parameters](../README.md#Illustration-of-segmentation-parameters) and [README/Segmentation-examples](../README.md#Segmentation-Examples). Please refer to the downloaded dataset for detailed examples. For further illustration, please refer to our paper.

**Note**: All audio files in the training and test set need to be fully annotated.
