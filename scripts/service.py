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
import numpy as np
import base64
import zipfile
import io
import plotly.figure_factory as ff
import plotly.graph_objects as go

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

def create_zip_in_memory_given_folder(folder):
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', compression=zipfile.ZIP_STORED) as zipf:
        for root, _, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=file)
    memory_file.seek(0)  # Reset file pointer to the beginning
    return memory_file

def create_zip_in_memory_given_uploaded_files(uploaded_files):
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', compression=zipfile.ZIP_STORED) as zipf:
        for uploaded_file in uploaded_files:
            zipf.writestr(uploaded_file.name, uploaded_file.read())
    memory_file.seek(0)  # Reset file pointer to the beginning
    return memory_file

def convert_df_to_datagrid_format(df):
    # Convert DataFrame columns to the format expected by DataGrid
    columns = [{'field': col, 'headerName': col, "width": 450 if col == "filename" else 70 } for col in df.columns]
    # Convert DataFrame rows to the format expected by DataGrid
    rows = df.to_dict(orient='records')
    # Add an 'id' field to each row if it doesn't already exist
    if 'id' not in df.columns:
        for i, row in enumerate(rows):
            row['id'] = i + 1
    return {'columns': columns, 'rows': rows}

def start_backend_service( flask_port, dataset_base_folder, model_base_folder ):
    subprocess.run( [ "python", os.path.join( script_dirname, "backend.py" ),
                      "--flask_port", str( flask_port ),
                      "--dataset_base_folder", str(dataset_base_folder), 
                      "--model_base_folder", str(model_base_folder)
                    ] )

@st.cache_resource
def init( flask_port, dataset_base_folder, model_base_folder ):
    print(datetime.now(), "backend service started!")
    t = threading.Thread( target = start_backend_service, 
                          args = ( flask_port, dataset_base_folder, model_base_folder ),
                          daemon = True
                        )
    t.start()
    ## wait until the backend service is ready 
    while True:
        try:
            status = requests.get(f"http://localhost:{flask_port}/status" ).json()
            print("backend is running ...")
            break
        except:
            time.sleep(1)


# Function to read the GIF file and encode it in base64
def get_gif_base64(path):
    import base64
    with open(path, "rb") as gif_file:
        gif_base64 = base64.b64encode(gif_file.read()).decode("utf-8")
    return gif_base64

def remove_streamlit_style():
    hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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
    if "queuing_gif" not in st.session_state:
        st.session_state["queuing_gif"] = get_gif_base64( script_dirname + "/assets/" + "queuing.gif"  )
    if "training_gif" not in st.session_state:
        st.session_state["training_gif"] = get_gif_base64( script_dirname + "/assets/" + "training.gif"  )
    if "rerun_now" not in st.session_state:
        st.session_state["rerun_now"] = False
    if "segmentation_status" not in st.session_state:
        st.session_state["segmentation_status"] = ""
    if "segmentation_df" not in st.session_state:
        st.session_state["segmentation_df"] = None
    
def list_models_available_for_finetuning(flask_port):
    return requests.post(f"http://localhost:{flask_port}/list-models-available-for-finetuning" ).json()["response"]

def list_models_available_for_inference(flask_port):
    return requests.post(f"http://localhost:{flask_port}/list-models-available-for-inference" ).json()["response"]

def list_models_being_trained(flask_port):
    return requests.post(f"http://localhost:{flask_port}/list-models-training-in-progress" ).json()["response"]

def list_all_models(flask_port):
    return requests.post(f"http://localhost:{flask_port}/list-all-models" ).json()["response"]

def segment_audio( url, model_name, audio_path, min_frequency = None, spec_time_step = None, channel_id = 0 ):
    response = requests.post( url, files = { "audio_file": open(audio_path, "rb").read() if isinstance(audio_path, str) else audio_path },
                                   data = { "model_name":model_name,
                                            "min_frequency":min_frequency,
                                            "spec_time_step":spec_time_step,
                                            "channel_id":channel_id
                                          })
    return response.json()

def submit_training_request( url, model_name, initial_model_name, uploaded_files, num_epochs = 3 ):
    memory_file = create_zip_in_memory_given_uploaded_files(uploaded_files)
    files = {'zip': memory_file }
    response = requests.post(url, files=files, data = { "model_name":model_name,
                                                        "initial_model_name":initial_model_name,
                                                        "num_epochs":num_epochs
                                                      })

    return response.json()

# Function to handle segmentation 
def handle_segmentation( flask_port, model_name, uploaded_files ):
    status_reporter = st.empty()
    
    if st.session_state["running_segmentation"]:
        st.session_state["running_segmentation"] = 0
        
        if len(uploaded_files) == 0:
            st.session_state["segmentation_status"] = "Please first upload audio files before click start." 
            status_reporter.write( st.session_state["segmentation_status"] )
        else:
            overall_prediction = {"filename":[],"onset":[], "offset":[], "cluster":[]}
            for count, uploaded_file in enumerate( uploaded_files ):
                audio_fname = uploaded_file.name
                st.session_state["segmentation_status"] = "Segmenting %s... (%d/%d)"%( audio_fname, count + 1, len(uploaded_files) )
                status_reporter.write( st.session_state["segmentation_status"] )
                prediction = segment_audio( f"http://localhost:{flask_port}/segment", 
                                            model_name,
                                            uploaded_file,
                                            min_frequency = st.session_state["min_frequency"],
                                            channel_id = st.session_state["channel_id"]
                )
                overall_prediction["filename"] += [ audio_fname ] * len( prediction["onset"] )
                overall_prediction["onset"] += prediction["onset"]
                overall_prediction["offset"] += prediction["offset"]
                overall_prediction["cluster"] += prediction["cluster"]
            st.session_state["segmentation_df"] = pd.DataFrame( overall_prediction )         
            
    status_reporter.write( st.session_state["segmentation_status"] ) 
    if st.session_state["segmentation_df"] is not None:
        df = st.session_state["segmentation_df"].copy()
        unique_fnames = list(set(df["filename"].tolist()))
        if len(unique_fnames) <= 1:
            del df["filename"]
        
        ### post-process the segmentation results
        if st.session_state["adobe_audition_compatible"]:
            
            Start_list = [ seconds_to_decimal( seconds ) for seconds in df["onset"].tolist() ] 
            Duration_list = [ seconds_to_decimal( end - start ) for start, end in zip( df["onset"].tolist(), df["offset"].tolist() )  ]
            Format_list = [ "decimal" ] * len(Start_list)
            Type_list = [ "Cue" ] * len(Start_list)
            Description_list = [ "" for _ in range(len(Start_list))]
            Name_list = list(df["cluster"])  

            if "filename" not in df:
                df = pd.DataFrame({
                    "\ufeffName":Name_list,
                    "Start":Start_list,
                    "Duration":Duration_list,
                    "Time Format":Format_list,
                    "Type":Type_list,
                    "Description":Description_list
                })
            else:
                df = pd.DataFrame({
                    "filename": df["filename"].tolist(),
                    "\ufeffName":Name_list,
                    "Start":Start_list,
                    "Duration":Duration_list,
                    "Time Format":Format_list,
                    "Type":Type_list,
                    "Description":Description_list
                })
            
        # Download button for CSV file
        csv = df.to_csv(index = False, sep="\t")
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="predicted-annotations.csv">Download CSV file</a>'
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

    
# Function to handle fine-tuning
def handle_fine_tuning(flask_port, model_name, initial_model_name, uploaded_files):
    status_reporter = st.empty()
    model_list = [ item["model_name"] for item in list_all_models( flask_port )]
    model_name =  model_name.lower().strip()
    if model_name == "":
        status_reporter.write("Error: The model name cannot be empty.")
    elif model_name in model_list:
        status_reporter.write("Error: The model name you entered already exists. Please choose a different name.")
    else:
        illegal_strings = list(set(re.findall("[^a-zA-Z0-9\-\_\.]+", model_name )))
        if len(illegal_strings) > 0:
            status_reporter.write("Error: '%s' not allowed in model name"%( " ".join( illegal_strings ) ))
        else:
            response = submit_training_request( f"http://localhost:{flask_port}/submit-training-request", model_name, initial_model_name, uploaded_files )
            status_reporter.write( json.dumps( response ) )
    
def display_segmentation_tab(flask_port):
    st.header("Segment")
    ## This is a hacky way to refresh the file uploader
    uploaded_files = st.file_uploader("Upload Audio Files" + " "*st.session_state["refresh_segmentation_tab"],  accept_multiple_files=True, type=["wav"])
    
    model_list = [ item["model_name"] for item in list_models_available_for_inference( flask_port )]
    model_name = st.selectbox("Choose WhisperSeg Model", model_list)

    cols = st.columns(3)
    with cols[0]:
        st.number_input('Audio Channel ID', value=0, key = "channel_id")
    with cols[1]:
        st.number_input('Minimum Frequency (Hz)', value= 0, key = "min_frequency", step = 1)
    with cols[2]:
        st.checkbox('Output CSV Adobe Audition Compatible', value=True, key = "adobe_audition_compatible")     
    
    cols = st.columns(7)
    with cols[0]:
        if st.button("Start", key="segment"):
            st.session_state["running_segmentation"] = 1
    if len(uploaded_files)>0:
        with cols[1]:
            if st.button("Refresh", key="refresh-segment"):
                st.session_state["refresh_segmentation_tab"] = np.random.choice(1000)  # add variant to the file uploader's name to refresh it
                st.session_state["segmentation_status"] = ""
                st.session_state["segmentation_df"] = None 
                st.rerun()
    handle_segmentation(flask_port, model_name, uploaded_files )

    ## this is important to refresh the widgets
    for _ in range(10):
        st.empty()  
        
def display_finetuning_tab(flask_port):
    st.header("Finetune")
    uploaded_files = st.file_uploader("Upload Training Dataset (Paired audio file and annotation csv/json, e.g., exp1_sound1.wav, exp1_sound1.csv, exp1_sound2.wav, exp1_sound2.csv ... For detailed data strcuture please refer to https://github.com/nianlonggu/WhisperSeg/blob/master/docs/DatasetProcessing.md)" + " "*st.session_state["refresh_finetuning_tab"], accept_multiple_files=True, type=["wav", "csv","json"], key="finetune_audio")
    
    model_list = [ item["model_name"] for item in list_models_available_for_finetuning( flask_port )]
    initial_model_name = st.selectbox("Select the model to use as the starting point for training", model_list, key="finetune_model")
    model_name = st.text_input("Name your new fine-tuned model using letters, numbers, or '-'. Avoid special characters like /\\?|}!. Ensure the name is unique and not used by existing models.")

    cols = st.columns(7)
    with cols[0]:
        if st.button("Start", key="finetune"):
            st.session_state["running_finetuning"] = 1
    if len(uploaded_files)>0:
        with cols[1]:
            if st.button("Refresh", key="refresh-finetune"):
                st.session_state["refresh_finetuning_tab"] =  np.random.choice(1000) 
                st.rerun()
    if st.session_state["running_finetuning"]:
        st.session_state["running_finetuning"] = 0
        handle_fine_tuning(flask_port, model_name, initial_model_name, uploaded_files)
    for _ in range(5):
        st.empty()

def send_rerun_signal():
    st.session_state["rerun_now"] = True

def display_model_list_tab(flask_port):
    st.header("Model List")
    st.session_state["all_model_list"] = list_all_models(flask_port)
        
    with elements("model-list"):
        # mui.Button("Refresh", variant = "text", onClick=lambda x:send_rerun_signal)
        with mui.List( sx={ "width": '100%', "maxWidth": "100%", "bgcolor": 'background.paper' } ): 
            mui.Divider()
            for idx in range(len( st.session_state["all_model_list"] )):
                with mui.ListItem():
                    mui.ListItemText( st.session_state["all_model_list"][idx]["model_name"], sx={"width":"20px", "height":"20px"} )
                    if st.session_state["all_model_list"][idx]["status"] == "queuing":
                        mui.Typography("Queuing")
                        with mui.ListItemIcon(): 
                            mui.Avatar(src="data:image/gif;base64,"+st.session_state["queuing_gif"],
                                   sx={"width":"20px", "height":"20px"}) 
                            
                    elif st.session_state["all_model_list"][idx]["status"] == "training":
                        mui.Typography("Training ETA: " + st.session_state["all_model_list"][idx].get("eta", "--:--:--"))
                        with mui.ListItemIcon():
                            mui.Avatar(src="data:image/gif;base64,"+st.session_state["training_gif"],
                                   sx={"width":"20px", "height":"20px"})
                            
                    elif st.session_state["all_model_list"][idx]["status"] == "ready": 
                        mui.Typography("Ready")
                        with mui.ListItemIcon(): 
                            mui.Icon( mui.icon.Done, sx={"width":"20px", "height":"20px"} )
                mui.Divider()
    if st.session_state["rerun_now"]:
        st.session_state["rerun_now"] = False
        st.rerun()
    ### This is important for refreshing the widgets
    for _ in range(10):
        st.empty()
                        
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
    remove_streamlit_style()
    
    # Define the layout
    st.title("WhisperSeg Application")
    # Create tabs
    tab_names = ["Segment", "Finetune", "Model List"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        display_segmentation_tab(args.backend_flask_port)
    with tabs[1]:
        display_finetuning_tab(args.backend_flask_port)
    with tabs[2]:
        display_model_list_tab(args.backend_flask_port)

    time.sleep(5)
    st.rerun()

if __name__ == "__main__":    
    main()