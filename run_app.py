import subprocess
import sys
import time
import webbrowser
import os

def check_dependencies():
    """Check if required packages are installed and install if needed"""
    required_packages = [
        "fastapi", "uvicorn", "pandas", "joblib", "pymodbus", 
        "gradio", "requests", "pydantic"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")

def run_api_server():
    """Start the FastAPI server in a separate process"""
    print("Starting FastAPI server...")
    api_process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    # Wait for the server to start
    time.sleep(2)
    return api_process

def run_ui():
    """Start the Gradio UI"""
    print("Starting Gradio UI...")
    ui_process = subprocess.Popen(
        [sys.executable, "ui.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return ui_process

def main():
    print("Setting up Crop Recommendation System...")
    check_dependencies()
    
    api_process = run_api_server()
    ui_process = run_ui()
    
    print("\n===================================================")
    print("Both API server and UI are now running!")
    print("Access the UI at: http://127.0.0.1:7860")
    print("API is running at: http://127.0.0.1:8010")
    print("\nPress Ctrl+C to shut down both servers")
    print("===================================================\n")
    
    # Open browser automatically
    webbrowser.open("http://127.0.0.1:7860")
    
    try:
        # Keep the script running until user interrupts
        api_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        api_process.terminate()
        ui_process.terminate()
        print("Servers shut down. Goodbye!")

if __name__ == "__main__":
    main()
