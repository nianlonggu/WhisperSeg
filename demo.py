import argparse
import base64
import io
import json
import librosa
import numpy as np
import os
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import requests
import shutil
import soundfile as sf
import streamlit as st
import threading
import time
from tqdm import tqdm
from uuid import uuid4
from copy import deepcopy
from datetime import datetime
from model import WhisperSegmenter, WhisperSegmenterFast
from datetime import datetime, timedelta
import threading
import time

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

@st.cache_resource
def load_segmenter():
    try:
        segmenter = WhisperSegmenterFast( args.model_path, device = args.device, device_ids = args.device_ids )
        print("The loaded model is the Ctranslated version.")
    except:
        segmenter = WhisperSegmenter( args.model_path, device = args.device, device_ids = args.device_ids )
        print("The loaded model is the original huggingface version.")
    return segmenter


parser = argparse.ArgumentParser(description='App external parameters')
parser.add_argument('--segment_config_path', default = "config/segment_config.json")
parser.add_argument('--model_path')
parser.add_argument('--device', default = "cuda")
parser.add_argument("--device_ids", help="a list of GPU ids", type = int, nargs = "+", default = [0,])
parser.add_argument("--species", help="main target species", default = None)
try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    os._exit(e.code)

    
def segment(segmenter, audio_data, channel_id, sr, min_frequency, spec_time_step, min_segment_length, eps, num_trials, adobe_audition_compatible, segment_results = {} ):
        
    ### load the audio
    audio, _ = librosa.load( io.BytesIO(audio_data), sr = sr, mono=False )
    if len(audio.shape) == 2:
        audio = audio[channel_id]
    audio_duration = len(audio) / sr
    segment_results["audio_duration"] = audio_duration
            
    ### segment the audio
    prediction = segmenter.segment(  audio, sr = sr, min_frequency = min_frequency, spec_time_step = spec_time_step,
                       min_segment_length = min_segment_length, eps = eps, num_trials = num_trials, status_monitor = segment_results )
    
    ### post-process the segmentation results
    if adobe_audition_compatible:
        Start_list = [ seconds_to_decimal( seconds ) for seconds in prediction["onset"] ] 
        Duration_list = [ seconds_to_decimal( end - start ) for start, end in zip( prediction["onset"], prediction["offset"] )  ]
        Format_list = [ "decimal" ] * len(Start_list)
        Type_list = [ "Cue" ] * len(Start_list)
        Description_list = [ "" for _ in range(len(Start_list))]
        Name_list = list(prediction["cluster"])  #[ "" for _ in range( len(Start_list) )  ]
        
        prediction = {
            "\ufeffName":Name_list,
            "Start":Start_list,
            "Duration":Duration_list,
            "Time Format":Format_list,
            "Type":Type_list,
            "Description":Description_list
        }
    segment_results['segmentation_df'] = pd.DataFrame( prediction )
    segment_results['is_done'] = True
    
    
# Callback function to update species
def update_hyperparameters():
    cfg = st.session_state["segment_config"].get( st.session_state["species"], {} )
    st.session_state['sr'] = cfg.get( "sr", 32000 )
    st.session_state['min_frequency'] = cfg.get( "min_frequency", 0 )
    st.session_state['spec_time_step'] = cfg.get( "spec_time_step", 0.0025 )
    st.session_state['min_segment_length'] = cfg.get( "min_segment_length", 0.01 )
    st.session_state['eps'] = cfg.get( "eps", 0.02 )
    st.session_state['num_trials'] = cfg.get( "num_trials", 3 )
    
def init_session_state():
    if "segment_config" not in st.session_state:
        st.session_state["segment_config"] = json.load(open(args.segment_config_path))
        st.session_state["species_list"] = sorted(list(st.session_state["segment_config"].keys())) + ["Other"]
        if args.species is not None and args.species in st.session_state["species_list"]:
           st.session_state["species_list"].remove(args.species)
           st.session_state["species_list"] = [args.species]  + st.session_state["species_list"]
    if "segmentation_csv_name" not in st.session_state:
        st.session_state["segmentation_csv_name"] = None
    if "segmentation_df" not in st.session_state:
        st.session_state["segmentation_df"] = None
    if "audio_data" not in st.session_state:
        st.session_state["audio_data"] = None
        
def refresh_button_calback():
    st.session_state["audio_data"] = None
    st.session_state["segmentation_df"] = None
    st.session_state["segmentation_csv_name"] = None

    
def main():
    
    init_session_state()
    segmenter = load_segmenter()
    
    st.title("Vocal Segmentation")
    
    # Sidebar for hyperparameters
    st.sidebar.title("Setting")
    st.sidebar.selectbox('Select Species', options=st.session_state["species_list"], key = "species", on_change = update_hyperparameters )
            
    st.markdown("""
<style>
    button.step-up {display: none;}
    button.step-down {display: none;}
    div[data-baseweb] {border-radius: 4px;}
</style>""", unsafe_allow_html=True)
    
    st.sidebar.number_input('Channel ID', value=0, key = "channel_id")
    st.sidebar.checkbox('Adobe Audition Compatible', value=True, key = "adobe_audition_compatible")     

    cfg = st.session_state["segment_config"].get( st.session_state["species_list"][0], {} )
    st.sidebar.number_input('Sample Rate (Hz)', value = cfg.get( "sr", None ), key = "sr", step = 1  )
    st.sidebar.number_input('Minimum Frequency (Hz)', value= cfg.get( "min_frequency", None ), key = "min_frequency", step = 1)
    st.sidebar.number_input('Spectrogram Time Step (s)', value= cfg.get( "spec_time_step", None ), key = "spec_time_step", step = 0.0001, format="%.4f")
    st.sidebar.number_input('Minimum Segment Length (s)', value= cfg.get( "min_segment_length", None ), key = "min_segment_length", step = 0.001, format="%.3f")
    st.sidebar.number_input('Epsilon (s)', value= cfg.get( "eps", None ), key = "eps", step = 0.001, format="%.3f")
    st.sidebar.number_input('Number of Trials', value= cfg.get( "num_trials", None ), key = "num_trials", step = 1)
                
    if st.session_state["audio_data"] is None:
        uploaded_file = st.file_uploader('Upload audio file', type=['wav', 'mp3'])
        if uploaded_file is not None:
            st.session_state["audio_data"] = uploaded_file.read()
            st.session_state["segmentation_csv_name"] = f"{uploaded_file.name[:-4]}_anno.csv"
            st.rerun()
    else:
        audio_data = st.session_state['audio_data']
        channel_id = st.session_state['channel_id']
        sr = st.session_state['sr']
        min_frequency = st.session_state['min_frequency']
        spec_time_step = st.session_state['spec_time_step']
        min_segment_length = st.session_state['min_segment_length']
        eps = st.session_state['eps'] 
        num_trials = st.session_state['num_trials']
        adobe_audition_compatible = st.session_state['adobe_audition_compatible']
        
        segment_results = { "progress":0, "is_done":False }
        t = threading.Thread( target = segment, args = ( segmenter, audio_data, channel_id, sr, min_frequency, spec_time_step, min_segment_length, eps, num_trials, adobe_audition_compatible, segment_results)  )
        t.start()
        
        progress_bar = st.empty()
        # progress_bar = progress_placeholder.progress(0 )
        eta_info = st.empty( )
        
        start_time = time.time()
        while True:
            current_progress = segment_results["progress"]
            current_time = time.time()
            if current_progress > 0:
                eta_time = np.round( ( current_time - start_time ) / current_progress * (100 - current_progress), 3)
                eta_time = str(timedelta( seconds = eta_time ))
            else:
                eta_time = "Inf"
            progress_bar.progress( current_progress )
            eta_info.text( f"ETA: {eta_time}" )
            if segment_results["is_done"]:
                break
            time.sleep(0.1)
        t.join()
        st.session_state["segmentation_df"] = segment_results["segmentation_df"]
        
    if st.session_state["segmentation_df"] is not None:
        progress_bar.empty()
        eta_info.empty()
        
        refresh_button = st.button("Refresh", key="refreshButton", on_click=refresh_button_calback)
        
        df = st.session_state["segmentation_df"]
        csv_name = st.session_state["segmentation_csv_name"]
        
        # Download button for CSV file
        csv = df.to_csv(index = False, sep="\t")
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{csv_name}">Download CSV file</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        columns = list( df.keys() )
        fig = go.Figure(
                 data=[go.Table(header=dict(values=columns),
                                cells=dict(values=[df[col_name] for col_name in columns  ]))
                     ] )
        fig.update_layout(
            height=800,
            margin=dict(l=0, r=0, b=0, t=0 )
        )
        st.plotly_chart(fig)
        

if __name__ == "__main__":
        
    main()
    
