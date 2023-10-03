# Run WhisperSeg as a Web Service

Runing WhisperSeg as a Web service make it possible to disentangle the environment of the WhisperSeg and the environment where this segmenting function is called. For example, we can set up a WhisperSeg segmenting service at one machine, and call the segmenting service in different working environment (Matlab, Webpage frontend, Jupyter Notebook) at different physical locations.

This enables an easy implementation of calling WhisperSeg in Matlab and is essential for setting up a Web page for automatic vocal segmentation.

## Step 1: Starting the segmenting service

In a terminal, go to the main folder of this repository, and run the following command:
```
python segment_service.py -flask_port 8050 -model_path nccratliri/whisperseg-large-ms-ct2 -device cuda
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

## define a function for segmentation
def call_segment_service( service_address, 
                          audio_file_path,
                          sr,
                          min_frequency,
                          spec_time_step,
                          min_segment_length,
                          eps,
                          num_trials
                        ):
    audio_file_base64_string = base64.b64encode( open(audio_file_path, 'rb').read()).decode('ASCII')
    response = requests.post( service_address,
                              data = json.dumps( {
                                  "audio_file_base64_string":audio_file_base64_string,
                                  "sr":sr,
                                  "min_frequency":min_frequency,
                                  "spec_time_step":spec_time_step,
                                  "min_segment_length":min_segment_length,
                                  "eps":eps,
                                  "num_trials":num_trials
                              } ),
                              headers = {"Content-Type": "application/json"}
                            )
    return response.json()

prediction = call_segment_service( "http://localhost:8050/segment", 
                          "data/example_subset/Zebra_finch/test_adults/zebra_finch_g17y2U-f00007.wav",                               
                          sr = 32000,
                          min_frequency = 0,
                          spec_time_step = 0.0025,
                          min_segment_length = 0.01,
                          eps = 0.02,
                          num_trials = 3
                        )
print(prediction)
```

    {'cluster': ['zebra_finch_0', 'zebra_finch_0', 'zebra_finch_0', 'zebra_finch_0', 'zebra_finch_0', 'zebra_finch_0', 'zebra_finch_0', 'zebra_finch_0', 'zebra_finch_0', 'zebra_finch_0', 'zebra_finch_0', 'zebra_finch_0', 'zebra_finch_0', 'zebra_finch_0', 'zebra_finch_0', 'zebra_finch_0', 'zebra_finch_0'], 'offset': [0.073, 0.447, 0.673, 0.83, 1.483, 1.882, 2.037, 2.643, 2.893, 3.063, 3.283, 3.742, 3.898, 4.523, 5.223, 5.393, 6.043], 'onset': [0.01, 0.38, 0.603, 0.758, 0.912, 1.813, 1.967, 2.073, 2.838, 2.982, 3.112, 3.668, 3.828, 3.953, 5.158, 5.323, 5.467]}


**Note:** 
1. Runing the above code does not require any further dependencies or load any models
2. The service_address is composed of **SEGMENTING_SERVER_IP_ADDRESS** + **":"** + **FLASK_PORT_NUMBER** + **"/segment"**. If the server is running in the local machine, then the SEGMENTING_SERVER_IP_ADDRESS is "http://localhost", otherwise, you will need to know the IP address of the server machine. 
3. The choice of the values for sr, min_frequency, spec_time_step, min_segment_length, eps and num_trials varies from dataset to dataset. To get the detailed setting of these parameters for different species, please refer to [README.md#Illustration-of-segmentation-parameters](../README.md#Illustration-of-segmentation-parameters)

### call the segmenting service in MATLAB:

First define a matlab function

```matlab
function response = call_segment_service(service_address, audio_file_path, sr, min_frequency, spec_time_step, min_segment_length, eps, num_trials)

    fileID = fopen(audio_file_path, 'r');
    fileData = fread(fileID, inf, 'uint8=>uint8');

    audio_file_base64_string = matlab.net.base64encode( fileData );
    data = struct('audio_file_base64_string', audio_file_base64_string, ...
                  "sr", sr, ...
                  "min_frequency", min_frequency, ...
                  "spec_time_step", spec_time_step, ...
                  "min_segment_length", min_segment_length, ...
                  "eps", eps, ...
                  "num_trials", num_trials );
    jsonData = jsonencode(data);

    options = weboptions( 'RequestMethod', 'POST', 'MediaType', 'application/json'  );
    response = webwrite(service_address, jsonData, options);

end
```

Then call the matlab function in MATLAB console:

```matlab
prediction = prediction = call_segment_service( 'http://localhost:8050/segment', '/Users/meilong/Downloads/zebra_finch_g17y2U-f00007.wav', 32000, 0, 0.0025, 0.01, 0.02, 3  )
disp(prediction)

```
