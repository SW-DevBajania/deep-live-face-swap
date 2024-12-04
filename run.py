import subprocess
import time
from modules import core

# Path to monitor.py (adjust the path as needed)
monitor_script = "monitor.py"

if __name__ == '__main__':
    # # Start monitor.py as a subprocess
    # monitor_process = subprocess.Popen(["python", monitor_script])
    # print("Monitor started...")

    # try:
        # Run the main logic in core.run()
    core.run()

    # finally:
    #     # Stop monitor.py when core.run() finishes
    #     print("Stopping monitor...")
    #     monitor_process.terminate()  # Gracefully terminate the process
    #     monitor_process.wait()  # Wait for the process to finish
    #     print("Monitor stopped.")