# Crop Recommendation System

A machine learning-powered application that recommends suitable crops based on soil and environmental parameters.

## Features

- Predict optimal crops based on input parameters
- Easy-to-use web interface built with Gradio
- Robust API backend with FastAPI
- Modular architecture for easy maintenance and extension

## Requirements

- Python 3.7+
- Required packages (automatically installed when running the application):
  - fastapi
  - uvicorn
  - pandas
  - joblib
  - pymodbus
  - gradio
  - requests
  - pydantic

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Run the application script which will check and install all dependencies:
   ```
   python run_app.py
   ```

## Usage

The application consists of two components:
- **API Server**: Handles the prediction logic and model operations
- **UI**: Provides a user-friendly interface to interact with the system

### Running the Application

Execute the main script:
```
python run_app.py
```

This will:
1. Check and install required dependencies
2. Start the FastAPI server
3. Launch the Gradio UI
4. Open the UI in your default web browser

### Accessing the Application

- Web UI: [http://127.0.0.1:7860](http://127.0.0.1:7860)
- API: [http://127.0.0.1:8010](http://127.0.0.1:8010)

### API Endpoints

- `/predict`: Submit soil and environmental parameters to get crop recommendations
- Additional endpoints documentation available at [http://127.0.0.1:8010/docs](http://127.0.0.1:8010/docs)

## Project Structure

- `run_app.py`: Main script to run the entire application
- `main.py`: FastAPI server implementation
- `ui.py`: Gradio UI implementation
- Models and other resources (stored in project directory)

## Stopping the Application

Press `Ctrl+C` in the terminal where you started the application to shut down both the API server and UI.
