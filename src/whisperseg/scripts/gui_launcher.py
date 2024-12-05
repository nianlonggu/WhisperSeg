import sys
import subprocess
from pathlib import Path
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    streamlit_args = []
    script_args = []
    
    # Separate streamlit and script arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith('--'):
            if sys.argv[i] in ['--server.port']:
                streamlit_args.extend([sys.argv[i], sys.argv[i+1]])
                i += 2
            else:
                script_args.extend([sys.argv[i], sys.argv[i+1]])
                i += 2
        else:
            i += 1

    cmd = [ "streamlit", "run", SCRIPT_DIR + "/gui.py" ] + streamlit_args + ["--"] + script_args
    subprocess.run(cmd)

if __name__ == "__main__":
    main()