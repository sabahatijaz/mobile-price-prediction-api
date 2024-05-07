# Mobile Price Prediction API

This project is an API for predicting mobile phone prices based on their specifications using machine learning models.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository_url>

2. Install dependencies:
cd mobile-price-prediction-api
pip install -r requirements.txt

3. Download trained models:
Download the trained models RF_trained_model.pkl and GB_trained_model.pkl and place them in the mobile-price-prediction-api directory.

4. Run the FastAPI server:
 ```uvicorn app:app --host 0.0.0.0 --port 8001


## Testing the API
You can test the API using cURL or Python Requests.
1. Using cURL:
curl -X 'POST' \
  'http://localhost:8001/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "battery_power": 1043.0,
    "blue": 1,
    "clock_speed": 1.8,
    "dual_sim": 1,
    "fc": 14.0,
    "four_g": 0,
    "int_memory": 5.0,
    "m_dep": 0.1,
    "mobile_wt": 193.0,
    "n_cores": 3.0,
    "pc": 16.0,
    "px_height": 226.0,
    "px_width": 1412.0,
    "ram": 3476.0,
    "sc_h": 12.0,
    "sc_w": 7.0,
    "talk_time": 2.0,
    "three_g": 0,
    "touch_screen": 1,
    "wifi": 0
  }'

2. Using Python Requests:
import requests

url = 'http://localhost:8001/predict'
data = {
    "battery_power": 1043.0,
    "blue": 1,
    "clock_speed": 1.8,
    "dual_sim": 1,
    "fc": 14.0,
    "four_g": 0,
    "int_memory": 5.0,
    "m_dep": 0.1,
    "mobile_wt": 193.0,
    "n_cores": 3.0,
    "pc": 16.0,
    "px_height": 226.0,
    "px_width": 1412.0,
    "ram": 3476.0,
    "sc_h": 12.0,
    "sc_w": 7.0,
    "talk_time": 2.0,
    "three_g": 0,
    "touch_screen": 1,
    "wifi": 0
}

response = requests.post(url, json=data)
print(response.json())


Additional Notes

    Ensure that the RF_trained_model.pkl and GB_trained_model.pkl files are present in the mobile-price-prediction-api directory before running the server.
    You may need to modify the repository_url placeholder in the setup instructions based on your actual repository URL.
    Feel free to add any additional notes or instructions specific to your project.

