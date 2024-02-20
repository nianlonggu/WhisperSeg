import streamlit as st
import time
import datetime

def calculate_eta(start_time, current_progress):
    """
    Calculate ETA based on the start time and current progress.
    :param start_time: The start time of the process.
    :param current_progress: The current progress (0 to 100).
    :return: ETA as a timedelta object.
    """
    if current_progress == 0:
        return "Calculating..."
    
    elapsed_time = time.time() - start_time
    total_time_estimate = elapsed_time / (current_progress / 100)
    remaining_time = total_time_estimate - elapsed_time
    return datetime.timedelta(seconds=round(remaining_time))

# Initialize progress bar and start time
progress_bar = st.progress(0)
eta_text = st.empty()
start_time = time.time()

# Example: Simulate a long process
for percent_complete in range(100):
    time.sleep(0.1)  # Simulate work being done
    progress_bar.progress(percent_complete + 1)
    
    # Calculate and display ETA
    eta = calculate_eta(start_time, percent_complete + 1)
    eta_text.text(f"ETA: {eta}")

st.write("Task Completed!")
