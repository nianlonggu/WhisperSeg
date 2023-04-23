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
        num_trials = request_info.get( "num_trials", 3 )
        min_segment_length= request_info.get( "min_segment_length", 0.02 )
        voting_time_step = request_info.get( "voting_time_step", 1.0 )
        voting_precision = request_info.get( "voting_precision", 0.001 )
        batch_size = request_info.get("batch_size", 16)
        max_length = request_info.get("max_length", 400)
        
        
        audio, _ = librosa.load( io.BytesIO(base64_string_to_bytes(audio_file_base64_string)), 
                                sr = segmenter.sr )
        prediction = segmenter.segment( audio, 
                                        num_trials = num_trials,
                                        min_segment_length = min_segment_length,
                                        voting_time_step = voting_time_step,
                                        voting_precision = voting_precision,
                                        batch_size = batch_size,
                                        max_length = max_length
                                      )
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


