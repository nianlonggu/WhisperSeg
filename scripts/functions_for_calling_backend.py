import requests, os, json
import zipfile
import io
def create_zip_in_memory_given_folder(folder):
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', compression=zipfile.ZIP_STORED) as zipf:
        for root, _, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=file)
    memory_file.seek(0)  # Reset file pointer to the beginning
    return memory_file

def train( server_address, folder, model_name, initial_model_name, num_epochs = 3 ):
    response = requests.post( 
         server_address + "/submit-training-request",
         files= {'zip': create_zip_in_memory_given_folder( folder ) }, 
         data = { "model_name":model_name,
              "initial_model_name":initial_model_name,
              "num_epochs":num_epochs
            }
    )
    return response.json()

def segment( server_address, audio_path, model_name, min_frequency = None, spec_time_step = None, channel_id = 0 ):
    response = requests.post( 
         server_address + "/segment",
         files= {'audio_file': open(audio_path, "rb") }, 
         data = { "model_name":model_name,
              "min_frequency":min_frequency,
              "spec_time_step":spec_time_step,
              "channel_id":channel_id
            }
    )
    return response.json()


