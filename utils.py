import os


RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP = 2

def create_if_not_exists(folder):
    if not os.path.exists(folder):
        os.makedirs( folder )
    return folder

def get_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups ]