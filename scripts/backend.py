import sys
import os
script_dirname = os.path.dirname(os.path.abspath(__file__))
script_parent_dirname = os.path.dirname(script_dirname)
sys.path.insert(0, script_parent_dirname)
runtime_dirname = os.getcwd()

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
from uuid import uuid4
import subprocess
import gc
import torch
import re
from pathlib import Path
import GPUtil
import zipfile

# Make Flask application
app = Flask(__name__)
CORS(app)
# maintain the returned order of keys!
app.json.sort_keys = False

def bytes_to_base64_string(f_bytes):
    return base64.b64encode(f_bytes).decode('ASCII')

def base64_string_to_bytes(base64_string):
    return base64.b64decode(base64_string)

def get_gpu_memory():
    try:
        # Get the list of available GPUs
        gpus = GPUtil.getGPUs()

        # Check if GPU 0 is available
        if len(gpus) > 0:
            # Get the free memory of GPU 0
            gpu_0 = gpus[0]
            memory_free = gpu_0.memoryFree
            memory_total = gpu_0.memoryTotal

            return memory_free, memory_total
        else:
            print("No GPUs found.")
            return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def list_models():
    global model_base_folder, training_request_queue, pretrained_models
    all_models = []
    for item in pretrained_models:
        all_models.append( {"model_name": item["model_name"],
                            "inference_model_path": item["inference_model_path"],
                            "finetune_model_path": item["finetune_model_path"],
                            "status":"ready"
                           } )

    ## sort model based on creation time
    candi_folders = [ os.path.basename(str(item)) for item in sorted(Path(model_base_folder).glob('*'), key=lambda x: x.stat().st_ctime)]
    for name in candi_folders:
        if os.path.isdir( os.path.join(model_base_folder, name) ) and name not in [ item["model_name"] for item in training_request_queue ]:
            inference_model_path = os.path.join( model_base_folder, name, "final_checkpoint_ct2" )
            finetune_model_path = os.path.join( model_base_folder, name, "final_checkpoint" )
            if not os.path.exists( inference_model_path ):
                inference_model_path = None
            if not os.path.exists( finetune_model_path ):
                finetune_model_path = None
            if finetune_model_path is not None or inference_model_path is not None:
                all_models.append( {"model_name": name,
                                    "inference_model_path": inference_model_path,
                                    "finetune_model_path": finetune_model_path,
                                    "status":"ready"
                                   } )
    for item in training_request_queue:
        all_models.append( { "model_name": item["model_name"],
                             "inference_model_path": None,
                             "finetune_model_path": None,
                             "status": item["status"]
                           } )

    for item in all_models:
        if item["status"] == "training":
            training_status_fname = os.path.join( model_base_folder, name, "status.json" )
            try:
                status_data = json.load( open( training_status_fname ) )
                eta = status_data["eta"]
                assert len(re.findall( "^\d+:\d+:\d+$", eta )) == 1
            except:
                eta = "--:--:--"
            item["eta"] = eta
                       
    return all_models

def periodic_list_models( model_information ):
    while True:
        model_information["all_models"] = list_models()
        time.sleep(1)

def release_gpu():
    gc.collect()
    torch.cuda.empty_cache()

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': "ready" }), 200

@app.route('/list-models-available-for-finetuning', methods=['POST'])
def list_models_available_for_finetuning():
    global model_information
    all_models = model_information["all_models"]
    models = [ { "model_name": item["model_name"], "status": item["status"], "eta":item.get("eta","--:--:--") } for item in all_models if item["finetune_model_path"] is not None and item["status"] == "ready" ]
    return jsonify({'response': models }), 200

@app.route('/list-models-available-for-inference', methods=['POST'])
def list_models_available_for_inference():
    all_models = model_information["all_models"]
    models = [ { "model_name": item["model_name"], "status": item["status"], "eta":item.get("eta","--:--:--") } for item in all_models if item["inference_model_path"] is not None and item["status"] == "ready" ]
    return jsonify({'response': models }), 200

@app.route('/list-models-training-in-progress', methods=['POST'])
def list_models_training_in_progress():
    all_models = model_information["all_models"]
    models = [ { "model_name": item["model_name"], "status": item["status"], "eta":item.get("eta","--:--:--") } for item in all_models if item["status"] != "ready" ]
    return jsonify({'response': models }), 200

@app.route('/list-all-models', methods=['POST'])
def list_all_models():
    all_models = model_information["all_models"]
    models = [ { "model_name": item["model_name"], "status": item["status"], "eta":item.get("eta","--:--:--") } for item in all_models ]
    return jsonify({'response': models }), 200

@app.route('/get-training-request-queue', methods=['POST'])
def get_training_request_queue():
    global training_request_queue
    return jsonify({'response': training_request_queue }), 200

@app.route('/submit-training-request', methods=['POST'])
def submit_training_request():
    global dataset_base_folder, training_request_queue, sem
    sem.acquire()
    try:
        error_msg = {}
        ### always lower-case the model_name and initial_model_name
        model_name = request.form.get('model_name', type=str, default=None)
        initial_model_name = request.form.get('initial_model_name', type=str, default=None)
        num_epochs = request.form.get('num_epochs', type=int, default=3)

        illegal_strings = list(set(re.findall("[^a-zA-Z0-9\-\_\.]+", model_name )))
        if len(illegal_strings) > 0:
            error_msg = {'error': 'Model name cannot contain special characters "%s"'%(" ".join( illegal_strings ))}
            assert False
        model_name = model_name.lower().strip()
        if model_name == "":
            error_msg = {'error': 'Model name cannot be empty' }
            assert False
        
        # first check if model_name exists
        all_existing_models = list_models()
        if model_name in [ item["model_name"] for item in all_existing_models ]:
            error_msg = {'error': 'Model name already exists'}
            assert False

        if initial_model_name is None:
            initial_model_name = "whisperseg-base"
        initial_model_name = initial_model_name.lower().strip()
        
        if initial_model_name not in [ item["model_name"] for item in all_existing_models if item["finetune_model_path"] is not None ]:
            error_msg = {'error': 'initial_model_name is not available for finetuning, call "list-models-available-for-finetuning" API to get the available model_name list'}
            assert False
        
        # upload the training dataset
        dataset_folder =  os.path.join( dataset_base_folder, model_name )
        if 'zip' not in request.files:
            error_msg = {'error': 'No training files are provided in the request'}
            assert False

        file = request.files['zip']
        if file:
            os.makedirs( dataset_folder, exist_ok= True )
            # Load the file into a BytesIO buffer
            memory_file = io.BytesIO(file.read())
        
            # Extract files from the zip archive
            with zipfile.ZipFile(memory_file, 'r') as zip_ref:
                zip_ref.extractall(dataset_folder)

        # update the training_request_queue 
        with threading.Lock():
            training_request_queue.append({
                "model_name": model_name,
                "initial_model_name":initial_model_name,
                "train_dataset_folder":dataset_folder,
                "num_epochs":num_epochs,
                "status":"queuing"
            })
    except:
        sem.release()
        return jsonify( error_msg ), 400
    sem.release()
    return jsonify({'message': 'Training'}), 200

@app.route('/segment', methods=['POST'])
def segment():
    global sem, running_segmenters, max_num_segmenters_in_ram
    sem.acquire()
    try:
        error_msg = {}
        model_name = request.form.get('model_name', type=str, default=None)
        if model_name is None:
            model_name = "whisperseg-base"
        model_name = model_name.lower().strip()
        min_frequency = request.form.get('min_frequency', type=int, default=None)
        spec_time_step = request.form.get('spec_time_step', type=float, default=None)
        channel_id = request.form.get('channel_id', type=int, default=0)
        num_trials = request.form.get('num_trials', type=int, default=1)

        if 'audio_file' not in request.files:
            error_msg = {'error': 'No audio_file is provided'}
            assert False
        audio_file = request.files['audio_file']

        all_models = list_models()
        model_path = None
        for item in all_models:
            if item["model_name"] == model_name and item["inference_model_path"] is not None and item["status"] == "ready":
                model_path = item["inference_model_path"]
                break
        if model_path is None:
            error_msg = {'error': 'model_name is not available for inference, call "list-models-available-for-inference" API to get the available model_name list'}
            assert False

        if model_name not in running_segmenters:
            if len(running_segmenters) >= max_num_segmenters_in_ram:
                model_name_list = sorted(list(running_segmenters.keys()), key = lambda x:running_segmenters[x]["usage"])
                ## remove the least used model if the capacity if reached
                del running_segmenters[model_name_list[0]]
                release_gpu()
            ## add the new segmenter
            running_segmenters[model_name] = {"usage":0,"segmenter":WhisperSegmenterFast( model_path, device = "cuda", device_ids = [0,] )}

        segmenter = running_segmenters[model_name]["segmenter"]
        running_segmenters[model_name]["usage"] += 1
        
        ## handling the multi-channel audio file
        audio, sr = librosa.load( io.BytesIO(audio_file.read().lstrip()), sr = None, mono=False )
        if len(audio.shape) == 2:
            audio = audio[ channel_id ]
        if len(audio.shape) != 1:
            error_msg = {'error': 'audio loading failed' }
            assert False
        
        prediction = segmenter.segment( audio, sr, min_frequency = min_frequency, spec_time_step = spec_time_step, num_trials = num_trials, batch_size = 8 )  

    except:
        sem.release()
        return jsonify({"onset":[],
                 "offset":[],
                 "cluster":[]
                }), 400
    sem.release()
    return jsonify( prediction ), 200

def run_training_script( training_request_queue ):
    global model_base_folder
    while True:
        if len(training_request_queue) > 0:
            print("Start training ...")
            with threading.Lock():
                training_request_queue[0]["status"] = "training"
            try:
                initial_model_name = training_request_queue[0]["initial_model_name"]
                initial_model_path = None
                for item in list_models():
                    if item["model_name"] == initial_model_name and item["finetune_model_path"] is not None and item["status"] == "ready":
                        initial_model_path = item["finetune_model_path"]
                        break
                assert initial_model_path is not None

                model_folder = os.path.join( model_base_folder, training_request_queue[0]["model_name"] )

                ## pause when GPU is currently busy for other tasks
                gpu_free_memory, gpu_total_memory = get_gpu_memory() 
                if gpu_free_memory is None or gpu_total_memory is None or gpu_free_memory / gpu_total_memory < 0.7:
                    print("Warning: GPU may be unavailable or insufficient for training. Pending ...")
                    time.sleep(60)
                    continue
                
                process_args = [ "python", os.path.join( script_parent_dirname, "train.py" ), 
                            "--initial_model_path", initial_model_path,
                            "--train_dataset_folder", training_request_queue[0]["train_dataset_folder"] + "/",
                            "--model_folder", model_folder,
                            "--max_num_epochs", str( training_request_queue[0]["num_epochs"] ),
                        ]
                subprocess.run( process_args )                
                
                print("Training finished.")
                training_request_queue.pop(0)
            except:
                print("Training error!")
                training_request_queue.pop(0)
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--flask_port", help="The port of the flask app.", default=8060, type=int)
    parser.add_argument("--dataset_base_folder", help="The folder that stores the uploaded dataset.", type=str)
    parser.add_argument("--model_base_folder", help="The folder that stores the finetuned models.", type=str)
    parser.add_argument("--max_num_segmenters_in_ram", default=1, type=int)
    
    args = parser.parse_args()

    dataset_base_folder = args.dataset_base_folder
    model_base_folder = args.model_base_folder
    os.makedirs( dataset_base_folder, exist_ok=True )
    os.makedirs( model_base_folder, exist_ok=True )
    max_num_segmenters_in_ram = args.max_num_segmenters_in_ram
    
    pretrained_models = [ 
                            { "model_name":"whisperseg-base", 
                            "inference_model_path": "nccratliri/whisperseg-base-animal-vad-ct2",
                            "finetune_model_path": "nccratliri/whisperseg-base-animal-vad"},
                            { "model_name":"whisperseg-large", 
                            "inference_model_path": "nccratliri/whisperseg-animal-vad-ct2",
                            "finetune_model_path": "nccratliri/whisperseg-animal-vad"},
                        ]
    training_request_queue = list()
    sem = threading.Semaphore()
    running_segmenters = {}
    model_information = { "all_models":[] }

    threading.Thread( target = run_training_script, args = ( training_request_queue, ), daemon = True ).start()
    threading.Thread( target = periodic_list_models, args = ( model_information, ), daemon = True ).start()
    
    print("Waiting for requests...")

    app.run(host='0.0.0.0', port=args.flask_port, threaded = True )


