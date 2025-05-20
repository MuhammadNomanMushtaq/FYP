import gradio as gr
import requests
import json

# API URL
API_URL = "http://127.0.0.1:8010"

def connect_sensor():
    """Connect to the sensor via the API"""
    try:
        response = requests.get(f"{API_URL}/connect")
        if response.status_code == 200:
            data = response.json()
            if data.get("connected", False):
                return "✅ Successfully connected to sensor"
            else:
                return "❌ Failed to connect to sensor"
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_sensor_values():
    """Get values from the sensor via the API"""
    try:
        response = requests.get(f"{API_URL}/get-values")
        print("HTTP Status:", response.status_code)
        print("Raw Response:", response.text)
        if response.status_code == 200:
            data = response.json()
            print("Sensor API Response:", data)
            # Ensure the backend returns these exact keys: "Temperature", "N", "P", "K", "pH", "EC", "Moisture"
            # If any key is missing or has a different name/case, the value will be 0.
            return (
                data.get("Temperature", 0),
                data.get("N", 0),
                data.get("P", 0),
                data.get("K", 0),
                data.get("pH", 0),
                data.get("EC", 0),
                data.get("Moisture", 0),
                "✅ Successfully fetched sensor values"
            )
        else:
            status_msg = f"❌ Error: {response.status_code} - {response.text}"
            print("API Error:", status_msg)
            return 0, 0, 0, 0, 0, 0, 0, status_msg
    except Exception as e:
        import traceback
        print("Request Exception:", str(e))
        traceback.print_exc()
        status_msg = f"❌ Error: {str(e)}"
        return 0, 0, 0, 0, 0, 0, 0, status_msg

def predict(temperature, n, p, k, ph, ec, moisture):
    """Make crop prediction via the API"""
    try:
        payload = {
            "Temperature": float(temperature),
            "N": float(n),
            "P": float(p),
            "K": float(k),
            "pH": float(ph),
            "EC": float(ec),
            "Moisture": float(moisture)
        }
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            recommended_crops = data.get("prediction", [])
            if recommended_crops:
                return f"Recommended crops: {', '.join(recommended_crops)}"
            else:
                return "No crop recommendations available"
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def predict_irrigation(temperature, n, p, k, ph, ec, moisture):
    """Make irrigation prediction via the API"""
    try:
        payload = {
            "Temperature": float(temperature),
            "N": float(n),
            "P": float(p),
            "K": float(k),
            "pH": float(ph),
            "EC": float(ec),
            "Moisture": float(moisture)
        }
        response = requests.post(f"{API_URL}/predict-irrigation", json=payload)
        if response.status_code == 200:
            data = response.json()
            irrigation = data.get("irrigation_prediction", [])
            if irrigation:
                return f"Irrigation prediction: {', '.join(map(str, irrigation))}"
            else:
                return "No irrigation prediction available"
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def predict_fertilizer(temperature, n, p, k, ph, ec, moisture):
    """Make fertilizer recommendation via the API"""
    try:
        payload = {
            "Temperature": float(temperature),
            "N": float(n),
            "P": float(p),
            "K": float(k),
            "pH": float(ph),
            "EC": float(ec),
            "Moisture": float(moisture)
        }
        response = requests.post(f"{API_URL}/predict-fertilizer", json=payload)
        if response.status_code == 200:
            data = response.json()
            fertilizer = data.get("fertilizer_recommendation", [])
            if fertilizer:
                return f"Fertilizer recommendation: {', '.join(map(str, fertilizer))}"
            else:
                return "No fertilizer recommendation available"
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create the Gradio Interface
with gr.Blocks(title="Crop Recommendation System") as demo:
    gr.Markdown("# Crop Recommendation System")
    gr.Markdown("Connect to sensor, get readings, and predict suitable crops, irrigation, and fertilizer recommendations")

    with gr.Row():
        with gr.Column():
            connect_btn = gr.Button("Connect to Sensor")
            connection_status = gr.Textbox(label="Connection Status", interactive=False)
            connect_btn.click(connect_sensor, inputs=[], outputs=[connection_status])

        with gr.Column():
            get_values_btn = gr.Button("Get Sensor Values")
            sensor_status = gr.Textbox(label="Sensor Status", interactive=False)

    with gr.Row():
        with gr.Column():
            temperature = gr.Number(label="Temperature (°C)", value=25.0)
            n_value = gr.Number(label="Nitrogen (N)", value=50)
            p_value = gr.Number(label="Phosphorus (P)", value=30)
            moisture_value = gr.Number(label="Moisture", value=20.0)
        with gr.Column():
            k_value = gr.Number(label="Potassium (K)", value=40)
            ph_value = gr.Number(label="pH", value=6.5)
            ec_value = gr.Number(label="Electrical Conductivity (EC)", value=0.5)

    get_values_btn.click(
        get_sensor_values,
        inputs=[],
        outputs=[temperature, n_value, p_value, k_value, ph_value, ec_value, moisture_value, sensor_status]
    )

    with gr.Row():
        predict_btn = gr.Button("Predict Suitable Crops", variant="primary")
        irrigation_btn = gr.Button("Predict Irrigation", variant="primary")
        fertilizer_btn = gr.Button("Fertilizer Recommendation", variant="primary")

    with gr.Row():
        prediction_result = gr.Textbox(label="Prediction Result", interactive=False)
        irrigation_result = gr.Textbox(label="Irrigation Prediction", interactive=False)
        fertilizer_result = gr.Textbox(label="Fertilizer Recommendation", interactive=False)

    predict_btn.click(
        predict,
        inputs=[temperature, n_value, p_value, k_value, ph_value, ec_value, moisture_value],
        outputs=[prediction_result]
    )
    irrigation_btn.click(
        predict_irrigation,
        inputs=[temperature, n_value, p_value, k_value, ph_value, ec_value, moisture_value],
        outputs=[irrigation_result]
    )
    fertilizer_btn.click(
        predict_fertilizer,
        inputs=[temperature, n_value, p_value, k_value, ph_value, ec_value, moisture_value],
        outputs=[fertilizer_result]
    )

    gr.Markdown("### Instructions:")
    gr.Markdown("""
    1. Click 'Connect to Sensor' to establish connection
    2. Click 'Get Sensor Values' to retrieve current readings
    3. Adjust values manually if needed
    4. Click 'Predict Suitable Crops', 'Predict Irrigation', or 'Fertilizer Recommendation' to get results
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
