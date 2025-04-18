# Run WhisperSeg as a Web Service

Runing WhisperSeg as a Web service make it possible to disentangle the environment of the WhisperSeg and the environment where this segmenting function is called. For example, we can set up a WhisperSeg segmenting service at one machine, and call the segmenting service in different working environment (Matlab, Webpage frontend, Jupyter Notebook) at different physical locations.

This enables an easy implementation of calling WhisperSeg in Matlab and is essential for setting up a Web page for automatic vocal segmentation.

## Step 1: Starting the segmenting service
In a terminal, go to the main folder of this repository, and run the following command:
```
python segment_service.py --flask_port 8050 --model_path nccratliri/whisperseg-large-ms-ct2 --device cuda
```

Illustration of the parameters:
* flask_port: the port that this service will keep listening to. Requests that are sent to this port will be handled by this service
* model_path: the path to the WhisperSeg model. This model can either be original huggingface model, e.g., nccratliri/whisperseg-large-ms, or CTranslate converted model, e.g., nccratliri/whisperseg-large-ms-ct2. If you choose to use the Ctranslate converted model, please make sure the converted model exists. If you have a different trained WhisperSeg checkpoint, replace "nccratliri/whisperseg-large-ms-ct2" with the path to the checkpoint.
* device: where to run the WhisperSeg. It can be cuda or cpu. By default we run the model on cuda

**Note**:
The terminal that runs this service needs to be kept open. On Linux system's terminal, one can first create a new screen and run the service in the created screen, to allow the service runing in the background.

## Step 2: Calling the segmenting service

### call the segmenting service in python:

For example, we are segmenting a zebra finch recording:


```python
import requests,json,base64
import pandas as pd
import librosa

## define a function for segmentation
def call_segment_service( service_address, 
                          audio_file_path,
                          sr = None,
                          channel_id = 0,
                          min_frequency=None,
                          spec_time_step=None,
                          min_segment_length=None,
                          eps=None,
                          num_trials=3,
                          adobe_audition_compatible=False
                        ):
    if sr is None:
        sr = librosa.get_samplerate(audio_file_path)
    audio_file_base64_string = base64.b64encode( open(audio_file_path, 'rb').read()).decode('ASCII')
    response = requests.post( service_address,
                              data = json.dumps( {
                                  "audio_file_base64_string":audio_file_base64_string,
                                  "channel_id":channel_id,
                                  "sr":sr,
                                  "min_frequency":min_frequency,
                                  "spec_time_step":spec_time_step,
                                  "min_segment_length":min_segment_length,
                                  "eps":eps,
                                  "num_trials":num_trials,
                                  "adobe_audition_compatible":adobe_audition_compatible
                              } ),
                              headers = {"Content-Type": "application/json"}
                            )
    return response.json()
```

**Note (Important):** 
1. Runing the above code does not require any further dependencies or load any models
2. The **service_address** is composed of **SEGMENTING_SERVER_IP_ADDRESS** + **":"** + **FLASK_PORT_NUMBER** + **"/segment"**. If the server is running in the local machine, then the SEGMENTING_SERVER_IP_ADDRESS is "http://localhost", otherwise, you will need to know the IP address of the server machine. 
3. **channel_id** is useful when the input audio file has multiple channels. In this case, channel_id can be used to specify which channel to segment. By default channel_id = 0, which means the first channel is used for segmentation.
4. The parameter **adobe_audition_compatible** is used to control the returned segmentation results format. If adobe_audition_compatible=1, the returned segmentation result is a dictionary that is comptible with Adobe Audition. This means after converting the dictionary to a Dataframe and then to a csv file, this csv file can be directly loaded into Adobe Audition. If adobe_audition_compatible=0, the segmentation result is a simple dictionary containing only "onset", "offset" and "cluster".

#### Get the Adobe Audition compitible segmentation results


```python
prediction = call_segment_service( "http://localhost:8050/segment", 
                          "../data/example_subset/Zebra_finch/test_adults/zebra_finch_g17y2U-f00007.wav",  
                          adobe_audition_compatible = True
                        )
## we can convert the returned dictionary into a pandas Dataframe
df = pd.DataFrame(prediction)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>﻿Name</th>
      <th>Start</th>
      <th>Duration</th>
      <th>Time Format</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>0:00.010</td>
      <td>0:00.063</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>0:00.380</td>
      <td>0:00.067</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>0:00.603</td>
      <td>0:00.070</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>0:00.758</td>
      <td>0:00.074</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>0:00.912</td>
      <td>0:00.571</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>0:01.812</td>
      <td>0:00.070</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>0:01.963</td>
      <td>0:00.074</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td>0:02.073</td>
      <td>0:00.570</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td>0:02.840</td>
      <td>0:00.053</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>0:02.982</td>
      <td>0:00.081</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>10</th>
      <td></td>
      <td>0:03.112</td>
      <td>0:00.171</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td>0:03.668</td>
      <td>0:00.074</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td>0:03.828</td>
      <td>0:00.070</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>13</th>
      <td></td>
      <td>0:03.953</td>
      <td>0:00.570</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>14</th>
      <td></td>
      <td>0:05.158</td>
      <td>0:00.065</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>15</th>
      <td></td>
      <td>0:05.323</td>
      <td>0:00.070</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
    <tr>
      <th>16</th>
      <td></td>
      <td>0:05.468</td>
      <td>0:00.575</td>
      <td>decimal</td>
      <td>Cue</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



We can save the df to the Adobe Audition compitible csv by (note: index = False, sep="\t" is necessary!):


```python
df.to_csv( "prediction_result.csv", index = False, sep="\t")
```

#### Get the simple segmentation results


```python
prediction = call_segment_service( "http://localhost:8050/segment", 
                          "../data/example_subset/Zebra_finch/test_adults/zebra_finch_g17y2U-f00007.wav",  
                          adobe_audition_compatible = False
                        )
## we can convert the returned dictionary into a pandas Dataframe
df = pd.DataFrame(prediction)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>onset</th>
      <th>offset</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.010</td>
      <td>0.073</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.380</td>
      <td>0.447</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.603</td>
      <td>0.673</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.758</td>
      <td>0.832</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.912</td>
      <td>1.483</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.812</td>
      <td>1.882</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.963</td>
      <td>2.037</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.073</td>
      <td>2.643</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.840</td>
      <td>2.893</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.982</td>
      <td>3.063</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3.112</td>
      <td>3.283</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3.668</td>
      <td>3.742</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3.828</td>
      <td>3.898</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3.953</td>
      <td>4.523</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5.158</td>
      <td>5.223</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5.323</td>
      <td>5.393</td>
      <td>vocal</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5.468</td>
      <td>6.043</td>
      <td>vocal</td>
    </tr>
  </tbody>
</table>
</div>



### call the segmenting service in MATLAB:

First define a matlab function

```matlab
function response = call_segment_service(service_address, audio_file_path, sr, channel_id, min_frequency, spec_time_step, min_segment_length, eps, num_trials, adobe_audition_compatible)

    fileID = fopen(audio_file_path, 'r');
    fileData = fread(fileID, inf, 'uint8=>uint8');

    audio_file_base64_string = matlab.net.base64encode( fileData );
    data = struct('audio_file_base64_string', audio_file_base64_string, ...
                  "channel_id", channel_id, ...
                  "sr", sr, ...
                  "min_frequency", min_frequency, ...
                  "spec_time_step", spec_time_step, ...
                  "min_segment_length", min_segment_length, ...
                  "eps", eps, ...
                  "num_trials", num_trials, ... 
                  "adobe_audition_compatible", adobe_audition_compatible );
    jsonData = jsonencode(data);

    options = weboptions( 'RequestMethod', 'POST', 'MediaType', 'application/json'  );
    response = webwrite(service_address, jsonData, options);

end
```

Then call the matlab function in MATLAB console:

```matlab
prediction = prediction = call_segment_service( 'http://localhost:8050/segment', '/Users/meilong/Downloads/zebra_finch_g17y2U-f00007.wav', 32000, 0, 0, 0.0025, 0.01, 0.02, 3, 0 )
disp(prediction)

```


```python

```
