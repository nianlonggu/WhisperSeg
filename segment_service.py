import argparse
import json
import requests
from datetime import datetime
from flask import Flask, jsonify, abort, make_response, request, Response
from flask_cors import CORS

from model import WhisperSegmenter, WhisperSegmenterFast
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time

import threading
import base64
import io


# Make Flask application
app = Flask(__name__)
CORS(app)
# maintain the returned order of keys!
app.json.sort_keys = False

def decimal_to_seconds( decimal_time ):
    splits = decimal_time.split(":")
    if len(splits) == 2:
        hours = 0
        minutes, seconds = splits
    elif len(splits) == 3:
        hours, minutes, seconds = splits
    else:
        assert False
    
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

def seconds_to_decimal( seconds ):
    hours = int(seconds // 3600)
    minutes = int(seconds // 60)
    seconds = seconds % 60
    
    if hours > 0:
        return "%d:%02d:%06.3f"%( hours, minutes, seconds )
    else:
        return "%d:%06.3f"%( minutes, seconds )


def bytes_to_base64_string(f_bytes):
    return base64.b64encode(f_bytes).decode('ASCII')

def base64_string_to_bytes(base64_string):
    return base64.b64decode(base64_string)

@app.route('/segment', methods=['POST'])
def segment():
    global args, segmenter, sem, segment_config

    sem.acquire()
    try:
        request_info = request.json
        audio_file_base64_string = request_info["audio_file_base64_string"]
        ### drop all the key-value pairs whose value is None, since we will determine the default value within this function.
        request_info = { k:v for k,v in request_info.items() if v is not None}
        
        if "species" in request_info and request_info["species"] in segment_config:
            cfg = segment_config[request_info["species"]]
            default_sr = cfg["sr"]
            default_min_frequency = cfg["min_frequency"]
            default_spec_time_step = cfg["spec_time_step"]
            default_min_segment_length = cfg["min_segment_length"]
            default_eps = cfg["eps"]
            default_num_trials = cfg["num_trials"]
        else:
            default_sr = 32000
            default_min_frequency = 0
            default_spec_time_step = 0.0025
            default_min_segment_length = 0.01
            default_eps = 0.02 
            default_num_trials = 3
            
        sr = request_info.get("sr", default_sr)
        min_frequency = request_info.get("min_frequency", default_min_frequency)
        spec_time_step = request_info.get( "spec_time_step", default_spec_time_step )
        min_segment_length = request_info.get( "min_segment_length", default_min_segment_length )
        eps = request_info.get( "eps", default_eps )
        num_trials = request_info.get( "num_trials", default_num_trials )
        
        channel_id = request_info.get( "channel_id", 0 )
        adobe_audition_compatible = request_info.get( "adobe_audition_compatible", False )
        
        audio, _ = librosa.load( io.BytesIO(base64_string_to_bytes(audio_file_base64_string)), 
                                 sr = sr, mono=False )
        ### for multiple channel audio, choose the desired channel
        if len(audio.shape) == 2:
            audio = audio[channel_id]
            
        prediction = segmenter.segment(  audio, sr = sr, min_frequency = min_frequency, spec_time_step = spec_time_step,
                       min_segment_length = min_segment_length, eps = eps,num_trials = num_trials, batch_size = args.batch_size )
    except:
        print("Segmentation Error! Returning an empty prediction ...")
        prediction = {
            "onset":[],
            "offset":[], 
            "cluster":[]
        }
        adobe_audition_compatible = False
        
    if adobe_audition_compatible:
        Start_list = [ seconds_to_decimal( seconds ) for seconds in prediction["onset"] ] 
        Duration_list = [ seconds_to_decimal( end - start ) for start, end in zip( prediction["onset"], prediction["offset"] )  ]
        Format_list = [ "decimal" ] * len(Start_list)
        Type_list = [ "Cue" ] * len(Start_list)
        Description_list = [ "" for _ in range(len(Start_list))]
        Name_list = [ "" for _ in range( len(Start_list) )  ]
        
        prediction = {
            "\ufeffName":Name_list,
            "Start":Start_list,
            "Duration":Duration_list,
            "Time Format":Format_list,
            "Type":Type_list,
            "Description":Description_list
        }
    
    sem.release()
    return jsonify(prediction), 201
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--flask_port", help="The port of the flask app.", default=8050, type=int)
    parser.add_argument("--model_path")
    parser.add_argument("--config_file_path", help="The file for recommended segment parameters of different species.", default="config/segment_config.json")
    parser.add_argument("--device", help="cpu or cuda", default = "cuda")
    parser.add_argument("--device_ids", help="a list of GPU ids", type = int, nargs = "+", default = [0,])
    parser.add_argument("--batch_size", default=8, type=int)
    args = parser.parse_args()
    
    try:
        segmenter = WhisperSegmenterFast( args.model_path, device = args.device, device_ids = args.device_ids )
        print("The loaded model is the Ctranslated version.")
    except:
        segmenter = WhisperSegmenter( args.model_path, device = args.device, device_ids = args.device_ids )
        print("The loaded model is the original huggingface version.")
    
    sem = threading.Semaphore()
    segment_config = json.load(open(args.config_file_path, "r"))
        
    print("Waiting for requests...")

    app.run(host='0.0.0.0', port=args.flask_port, threaded = True )


