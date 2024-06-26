import sys
import os
script_dirname = os.path.dirname(os.path.abspath(__file__))
from streamlit_elements import elements, mui, dashboard
import re
import streamlit as st
import pandas as pd
from datetime import datetime
import threading
import time
import requests, json
import subprocess
import argparse

def start_backend_service( flask_port, dataset_base_folder, model_base_folder ):
    subprocess.run( [ "python", os.path.join( script_dirname, "backend_service.py" ),
                      "--flask_port", str( flask_port ),
                      "--dataset_base_folder", str(dataset_base_folder),
                      "--model_base_folder", str(model_base_folder)
                    ] )

@st.cache_resource
def init( flask_port, dataset_base_folder, model_base_folder ):
    print(datetime.now(), "backend service started!")
    t = threading.Thread( target=start_backend_service, args = ( flask_port, dataset_base_folder, model_base_folder ) )
    t.daemon = True
    t.start()

def init_varaiables():
    if "refresh_segmentation_tab" not in st.session_state:
        st.session_state["refresh_segmentation_tab"] = 0
    if "running_segmentation" not in st.session_state:
        st.session_state["running_segmentation"] = 0
    if "refresh_finetuning_tab" not in st.session_state:
        st.session_state["refresh_finetuning_tab"] = 0
    if "running_finetuning" not in st.session_state:
        st.session_state["running_finetuning"] = 0
    if "all_model_list" not in st.session_state:
        st.session_state["all_model_list"] = []

def list_models_available_for_finetuning(flask_port):
    return requests.post(f"http://localhost:{flask_port}/list-models-available-for-finetuning" ).json()["response"]

def list_models_available_for_inference(flask_port):
    return requests.post(f"http://localhost:{flask_port}/list-models-available-for-inference" ).json()["response"]

def list_models_being_trained(flask_port):
    return requests.post(f"http://localhost:{flask_port}/list-models-training-in-progress" ).json()["response"]

def list_all_models(flask_port):
    return requests.post(f"http://localhost:{flask_port}/list-all-models" ).json()["response"]

def segment_audio( url, model_name, audio_path, min_frequency = None, spec_time_step = None ):
    response = requests.post( url, files = { "audio_file": open(audio_path, "rb").read() if isinstance(audio_path, str) else audio_path },
                                   data = { "model_name":model_name,
                                            "min_frequency":min_frequency,
                                            "spec_time_step":spec_time_step
                                          })
    return response.json()

def submit_training_request( url, model_name, inital_model_name, uploaded_files, num_epochs = 3 ):
    files = [ ('files', (uploaded_file.name, uploaded_file)) for uploaded_file in uploaded_files ]
    response = requests.post(url, files=files, data = { "model_name":model_name,
                                                        "inital_model_name":inital_model_name,
                                                        "num_epochs":num_epochs
                                                      })

    return response.json()

# Function to handle segmentation
def handle_segmentation( flask_port, model_name, uploaded_files ):
    status_reporter = st.empty()
    if len(uploaded_files) == 0:
        status_reporter.write("Please first upload audio files before click start.")
    else:
        overall_prediction = {"filename":[],"onset":[], "offset":[], "cluster":[]}
        for count, uploaded_file in enumerate( uploaded_files ):
            audio_fname = uploaded_file.name
            status_reporter.write( "Segmenting %s... (%d/%d)"%( audio_fname, count + 1, len(uploaded_files) ) )
            prediction = segment_audio( f"http://localhost:{flask_port}/segment", 
                                        model_name,
                                        uploaded_file                             
            )
            overall_prediction["filename"] += [ audio_fname ] * len( prediction["onset"] )
            overall_prediction["onset"] += prediction["onset"]
            overall_prediction["offset"] += prediction["offset"]
            overall_prediction["cluster"] += prediction["cluster"]
        df = pd.DataFrame( overall_prediction )
        st.dataframe( df )
    
    
# Function to handle fine-tuning
def handle_fine_tuning(flask_port, model_name, inital_model_name, uploaded_files):
    status_reporter = st.empty()
    model_list = [ item["model_name"] for item in list_all_models( flask_port )]
    model_name =  model_name.lower().strip()
    if model_name == "":
        status_reporter.write("Error: The model name cannot be empty.")
    elif model_name in model_list:
        status_reporter.write("Error: The model name you entered already exists. Please choose a different name.")
    else:
        illegal_strings = list(set(re.findall("[^a-zA-Z0-9\-\_]+", model_name )))
        if len(illegal_strings) > 0:
            status_reporter.write("Error: '%s' not allowed in model name"%( " ".join( illegal_strings ) ))
        else:
            response = submit_training_request( f"http://localhost:{flask_port}/submit-training-request", model_name, inital_model_name, uploaded_files )
            status_reporter.write( json.dumps( response ) )
    

def display_segmentation_tab(flask_port):
    st.header("Segment")
    ## This is a hacky way to refresh the file uploader
    uploaded_files = st.file_uploader("Upload Audio Files" + " "*st.session_state["refresh_segmentation_tab"], accept_multiple_files=True, type=["wav"])
    
    model_list = [ item["model_name"] for item in list_models_available_for_inference( flask_port )]
    model_name = st.selectbox("Choose WhisperSeg Model", model_list)

    cols = st.columns(7)
    with cols[0]:
        if st.button("Start", key="segment"):
            st.session_state["running_segmentation"] = 1
    if len(uploaded_files)>0:
        with cols[1]:
            if st.button("Refresh", key="refresh-segment"):
                st.session_state["refresh_segmentation_tab"] =  1 - st.session_state["refresh_segmentation_tab"]
                st.rerun()
    if st.session_state["running_segmentation"]:
        st.session_state["running_segmentation"] = 0
        handle_segmentation(flask_port, model_name, uploaded_files )
        
def display_finetuning_tab(flask_port):
    st.header("Finetune")
    uploaded_files = st.file_uploader("Upload Training Dataset (Paired audio file and annotation csv/json, e.g., exp1_sound1.wav, exp1_sound1.csv, exp1_sound2.wav, exp1_sound2.csv ... For detailed data strcuture please refer to https://github.com/nianlonggu/WhisperSeg/blob/master/docs/DatasetProcessing.md)" + " "*st.session_state["refresh_finetuning_tab"], accept_multiple_files=True, type=["wav", "csv","json"], key="finetune_audio")
    
    model_list = [ item["model_name"] for item in list_models_available_for_finetuning( flask_port )]
    inital_model_name = st.selectbox("Select the model to use as the starting point for training", model_list, key="finetune_model")
    model_name = st.text_input("Name your new fine-tuned model using letters, numbers, or '-'. Avoid special characters like /\\?|}!. Ensure the name is unique and not used by existing models.")

    cols = st.columns(7)
    with cols[0]:
        if st.button("Start", key="finetune"):
            st.session_state["running_finetuning"] = 1
    if len(uploaded_files)>0:
        with cols[1]:
            if st.button("Refresh", key="refresh-finetune"):
                st.session_state["refresh_finetuning_tab"] =  1 - st.session_state["refresh_finetuning_tab"]
                st.rerun()
    if st.session_state["running_finetuning"]:
        st.session_state["running_finetuning"] = 0
        handle_fine_tuning(flask_port, model_name, inital_model_name, uploaded_files)

def display_model_list_tab(flask_port):
    st.header("Model List")
    if st.button("Refresh"):
        st.session_state["all_model_list"] = list_all_models(flask_port)    
    st.session_state["all_model_list"] = list_all_models(flask_port)
    status_symbols = {
        "ready": "✅",
        "queuing": "🕒",
        "training": "🔄"
    }

    # Create DataFrame
    model_df = pd.DataFrame( st.session_state["all_model_list"] )
    model_df[''] =  model_df['status'].map(status_symbols) 
    
    # # Display model list DataFrame
    st.dataframe(model_df.style.format({'status_symbol': lambda x: f'{x}'}), height=600)


def main():
    parser = argparse.ArgumentParser(description='App external parameters')
    parser.add_argument("--backend_flask_port", help="The port of the backend flask app.", default=8060, type=int)
    parser.add_argument("--backend_dataset_base_folder", help="The folder that stores the uploaded dataset.", type=str)
    parser.add_argument("--backend_model_base_folder", help="The folder that stores the finetuned models.", type=str)
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently streamlit prevents the program from exiting normally
        # so we have to do a hard exit.
        os._exit(e.code)

    init( args.backend_flask_port, args.backend_dataset_base_folder, args.backend_model_base_folder )
    init_varaiables()
    
    # Define the layout
    st.title("WhisperSeg Application")
    # Create tabs
    tab_names = ["Segment", "Finetune", "Model List"]
    tabs = st.tabs(tab_names)
    # Segment tab
    with tabs[0]:
        display_segmentation_tab(args.backend_flask_port)
    # Fine-tune tab
    with tabs[1]:
        display_finetuning_tab(args.backend_flask_port)
    with tabs[2]:
        display_model_list_tab(args.backend_flask_port)
            

if __name__ == "__main__":
    main()