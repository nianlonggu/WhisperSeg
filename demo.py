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
        segmenter = WhisperSegmenterFast( args.model_path, device = args.device )
        print("The loaded model is the Ctranslated version.")
    except:
        segmenter = WhisperSegmenter( args.model_path, device = args.device )
        print("The loaded model is the original huggingface version.")
    return segmenter


parser = argparse.ArgumentParser(description='App external parameters')
parser.add_argument('--segment_config_path', default = "config/segment_config.json")
parser.add_argument('--model_path')
parser.add_argument('--device', default = "cuda")
try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    os._exit(e.code)

def segment( audio_file_handle, audio_info, segmenter ):
    channel_id = st.session_state['channel_id']
    sr = st.session_state['sr']
    min_frequency = st.session_state['min_frequency']
    spec_time_step = st.session_state['spec_time_step']
    min_segment_length = st.session_state['min_segment_length']
    eps = st.session_state['eps'] 
    num_trials = st.session_state['num_trials']
    adobe_audition_compatible = st.session_state['adobe_audition_compatible']
    
    # print("\nTime:", datetime.now())   
    # print({"channel_id":channel_id, "sr":sr, "min_frequency":min_frequency, "spec_time_step":spec_time_step,
    #        "min_segment_length":min_segment_length, "eps":eps, "num_trials":num_trials, "adobe_audition_compatible":adobe_audition_compatible
    #       })
    
    ### load the audio
    audio, _ = librosa.load( audio_file_handle, sr = sr, mono=False )
    if len(audio.shape) == 2:
        audio = audio[channel_id]
    audio_duration = len(audio) / sr
    audio_info.text( "Length of the audio: %.2f s\nThis may take around %d minutes to segment ..." %( len(audio)/sr, int( np.ceil( audio_duration / 3 / 60 ) ) ) )
            
    ### segment the audio
    prediction = segmenter.segment(  audio, sr = sr, min_frequency = min_frequency, spec_time_step = spec_time_step,
                       min_segment_length = min_segment_length, eps = eps, num_trials = num_trials )
    
    ### post-process the segmentation results
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
    return pd.DataFrame( prediction )
    
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
        #if "meerkat" in st.session_state["species_list"]:
        #    st.session_state["species_list"].remove("meerkat")
        #    st.session_state["species_list"] = ["meerkat"]  + st.session_state["species_list"]
    
    if "species_changed" not in st.session_state:
        st.session_state['species_changed'] = False
    
def main():
    
    init_session_state()
    segmenter = load_segmenter()
    
    st.title("Vocal Segmentation")
    st.write('Upload an audio file to process')
    
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

    if True:
        cfg = st.session_state["segment_config"].get( st.session_state["species_list"][0], {} )
        st.sidebar.number_input('Sample Rate (Hz)', value = cfg.get( "sr", None ), key = "sr", step = 1  )
        st.sidebar.number_input('Minimum Frequency (Hz)', value= cfg.get( "min_frequency", None ), key = "min_frequency", step = 1)
        st.sidebar.number_input('Spectrogram Time Step (s)', value= cfg.get( "spec_time_step", None ), key = "spec_time_step", step = 0.0001, format="%.4f")
        st.sidebar.number_input('Minimum Segment Length (s)', value= cfg.get( "min_segment_length", None ), key = "min_segment_length", step = 0.001, format="%.3f")
        st.sidebar.number_input('Epsilon (s)', value= cfg.get( "eps", None ), key = "eps", step = 0.001, format="%.3f")
        st.sidebar.number_input('Number of Trials', value= cfg.get( "num_trials", None ), key = "num_trials", step = 1)
        
    # File upload
    uploaded_file = st.file_uploader('Upload audio file', type=['wav', 'mp3'])
    audio_info = st.empty()
    
    if uploaded_file is not None:
        
        df = segment( uploaded_file, audio_info, segmenter )

        # Download button for CSV file
        csv = df.to_csv(index = False, sep="\t")
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{uploaded_file.name[:-4]}_anno.csv">Download CSV file</a>'
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
    
