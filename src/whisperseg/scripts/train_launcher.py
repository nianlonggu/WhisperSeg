import sys
import subprocess
from pathlib import Path
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    script_path = "whisperseg.scripts.train"  # Use module path instead of file path
    torchrun_args = []
    script_args = []
    
    # Separate torchrun and script arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith('--'):
            if sys.argv[i] in ['--nproc_per_node', '--nnodes']:
                torchrun_args.extend([sys.argv[i], sys.argv[i+1]])
                i += 2
            else:
                script_args.extend([sys.argv[i], sys.argv[i+1]])
                i += 2
        else:
            i += 1
            
    if len(torchrun_args) == 0:
        cmd = ["python", SCRIPT_DIR + "/train.py"] + script_args
    else:
        cmd = ["torchrun"] + torchrun_args + ["-m", script_path] + script_args
    subprocess.run(cmd)

if __name__ == "__main__":
    main()