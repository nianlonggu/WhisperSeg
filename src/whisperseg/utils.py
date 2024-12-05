import os
import zipfile


RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP = 2

def create_if_not_exists(folder):
    if not os.path.exists(folder):
        os.makedirs( folder )
    return folder

def get_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups ]


def create_zip_file(folder_path, zip_file_path):
    all_files = list(os.walk(folder_path))
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_STORED) as zf:
        for root, dirs, files in all_files:
            for file in files:
                file_path = os.path.join(root, file)
                zf.write(file_path, os.path.relpath(file_path, folder_path))
    return zip_file_path