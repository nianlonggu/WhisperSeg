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


def bytes_to_base64_string(f_bytes):
    return base64.b64encode(f_bytes).decode('ASCII')

def base64_string_to_bytes(base64_string):
    return base64.b64decode(base64_string)

@app.route('/segment', methods=['POST'])
def segment():
    global args, segmenter, sem

    sem.acquire()
    try:
        request_info = request.json
        audio_file_base64_string = request_info["audio_file_base64_string"]
        sr = request_info.get("sr", 32000)
        min_frequency = request_info.get("min_frequency", 0)
        spec_time_step = request_info.get( "spec_time_step", 0.0025 )
        min_segment_length = request_info.get( "min_segment_length", 0.01 )
        eps = request_info.get( "eps", 0.02 )
        num_trials = request_info.get( "num_trials", 3 )
        
        audio, _ = librosa.load( io.BytesIO(base64_string_to_bytes(audio_file_base64_string)), 
                                 sr = sr )
        
        prediction = segmenter.segment(  audio, sr = sr, min_frequency = min_frequency, spec_time_step = spec_time_step,
                       min_segment_length = min_segment_length, eps = eps,num_trials = num_trials )
        
    except:
        print("Segmentation Error! Returning an empty prediction ...")
        prediction = {
            "onset":[],
            "offset":[], 
            "cluster":[]
        }
    
    sem.release()
    return jsonify(prediction), 201
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-flask_port", help="The port of the flask app.", default=8050, type=int)
    parser.add_argument("-model_path")
    parser.add_argument("-device", help="cpu or cuda", default = "cuda")
    args = parser.parse_args()
    
    try:
        segmenter = WhisperSegmenterFast( args.model_path, device = args.device )
        print("The loaded model is the Ctranslated version.")
    except:
        segmenter = WhisperSegmenter( args.model_path, device = args.device )
        print("The loaded model is the original huggingface version.")
    
    sem = threading.Semaphore()
        
    print("Waiting for requests...")

    app.run(host='0.0.0.0', port=args.flask_port, threaded = True )


