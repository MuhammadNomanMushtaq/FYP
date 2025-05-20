import joblib
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import struct
# Modbus sensor configuration
from pymodbus.client import ModbusSerialClient
import pandas as pd

# Load the trained model (change to GaussianNB and file name if needed)
model = joblib.load('gnb_model.pkl')  # Make sure to save your GaussianNB model as 'gnb_model.pkl'
irrig_model = joblib.load('irrig_model.pkl')
fert_model = joblib.load('fert_model.pkl')
sensor_client = None

app = FastAPI()
SENSOR_CONNECTED = False

class PredictionInput(BaseModel):
    Temperature: float
    N: float
    P: float
    K: float
    pH: float  # pH value
    EC: float  # Electrical Conductivity
    Moisture: float  # Add Moisture feature

    class Config:
        schema_extra = {
            "example": {
                "Temperature": 25.0,
                "N": 50,
                "P": 30,
                "K": 40,
                "pH": 6.5,
                "EC": 0.5,
                "Moisture": 20.0
            }
        }

CROP_LABELS = ['Wheat', 'Rice', 'Cotton', 'Maize', 'Sugarcane', 'Barley', 'Chickpea', 'Lentil', 'Mustard', 'Sesame', 'Sunflower']

def read_float(client, address):
    try:
        response = client.read_holding_registers(address=address, count=2, slave=1)
        if response.isError():
            return float('nan')
        raw_value = struct.unpack('>f', struct.pack('>HH', *response.registers))[0]
        return raw_value
    except Exception:
        return float('nan')

def read_scaled_int(client, address, factor=10):
    try:
        response = client.read_holding_registers(address=address, count=1, slave=1)
        if not response.isError():
            return response.registers[0] / factor
        return float('nan')
    except Exception:
        return float('nan')

@app.get("/connect")
def connect_sensor():
    try:
        sensor_client = ModbusSerialClient(
            port='COM3',
            baudrate=4800,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=2
        )
        if sensor_client.connect():
            SENSOR_CONNECTED = True
            
            return JSONResponse(content={"connected": True}, status_code=200)
        else:
            sensor_client.close()
            return JSONResponse(content={"connected": False}, status_code=503)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/get-values")
def get_sensor_values():
    import time
    try:
        sensor_client = ModbusSerialClient(
            port='COM3',
            baudrate=4800,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=2
        )
        if not sensor_client.connect():
            return JSONResponse(content={"error": "Could not connect to sensor"}, status_code=503)
        try:
            # Read all values first, then build the dictionary
            temp = read_scaled_int(sensor_client, 0x00)
            ec = read_float(sensor_client, 0x01)
            ph = read_scaled_int(sensor_client, 0x02)
            n = read_float(sensor_client, 0x03)
            p = read_float(sensor_client, 0x04)
            k = read_float(sensor_client, 0x05)
            moisture = read_scaled_int(sensor_client, 0x06)  # moisture last

            sensor_data = {
                "Temperature": temp,
                "N": n,
                "P": p,
                "K": k,
                "EC": ec,
                "pH": ph,
                "Moisture": moisture
            }

            return JSONResponse(
                content=sensor_data,
                status_code=200
            )
        finally:
            sensor_client.close()
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "Crop prediction API is running."}

@app.post("/predict")
def predict(input: PredictionInput):
    input_df = pd.DataFrame([input.dict()])
    prediction = model.predict(input_df)
    mapped = [CROP_LABELS[p] for p in prediction]
    return {"prediction": mapped}

@app.post("/predict-irrigation")
def predict_irrigation(input: PredictionInput):
    # Only use required features for irrigation model
    # Predict crop label if not provided
    crop_val = None
    if hasattr(input, "Crop"):
        crop_val = input.Crop
    else:
        crop_pred = model.predict(pd.DataFrame([input.dict()]))
        crop_val = crop_pred[0]
    input_df = pd.DataFrame([{
        "Temperature": input.Temperature,
        "pH": input.pH,
        "EC": input.EC,
        "Crop": crop_val
    }])
    prediction = irrig_model.predict(input_df)
    return {"irrigation_prediction": prediction.tolist()}

@app.post("/predict-fertilizer")
def predict_fertilizer(input: PredictionInput):
    # Only use required features for fertilizer model
    # Predict crop label if not provided
    crop_val = None
    if hasattr(input, "Crop"):
        crop_val = input.Crop
    else:
        crop_pred = model.predict(pd.DataFrame([input.dict()]))
        crop_val = crop_pred[0]
    input_df = pd.DataFrame([{
        "Temperature": input.Temperature,
        "pH": input.pH,
        "EC": input.EC,
        "Moisture": input.Moisture,
        "Crop": crop_val
    }])
    prediction = fert_model.predict(input_df)
    return {"fertilizer_recommendation": prediction.tolist()}

# input_data = pd.DataFrame([{
#     'N': 50,
#     'P': 30,
#     'K': 40,
#     'average_NPK': (50+30+40)/3,
#     'fertility_label': 1,  # example label
#     'temperature': 25.0,
#     'PH': 6.5
# }])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8010, reload=True)

