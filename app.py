from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import joblib
from EDA import EDA
from preprocess import preprocess, feature_engineering,select_features
from GradientBoostingClassifier import GradientBoostingclassifier
from RandomForestClassifier import RandomForestclassifier
import pandas as pd
app = FastAPI()


class Device(BaseModel):
    battery_power: float
    blue: int
    clock_speed: float
    dual_sim: int
    fc: float
    four_g: int
    int_memory: float
    m_dep: float
    mobile_wt: float
    n_cores: float
    pc: float
    px_height: float
    px_width: float
    ram: float
    sc_h: float
    sc_w: float
    talk_time: float
    three_g: int
    touch_screen: int
    wifi: int

    
def save_selected_features(selected_features, filename):
    with open(filename, 'w') as file:
        for feature in selected_features:
            file.write(feature + '\n')


def load_selected_features(filename):
    selected_features = []
    with open(filename, 'r') as file:
        for line in file:
            # Remove newline character and any leading/trailing whitespace
            feature = line.strip()
            selected_features.append(feature)
    return selected_features
# Define the endpoint for price prediction


@app.post('/predict')
async def predict_price(device_specs: Device):
    try:
        # Load the trained machine learning model
        RF_model = joblib.load('RF_trained_model.pkl')
        GB_model = joblib.load('GB_trained_model.pkl')
        
        # Load selected features
        selected_features = load_selected_features("selected_features.txt")
        
        # Remove unwanted features
        # selected_features.remove('pixel_density')
        selected_features.remove('price_range')
        # Create a dictionary from device_specs attributes
        device_dict = device_specs.dict()

        # Check if 'px_height' and 'px_width' are present in device_specs
        if 'px_height' in device_specs.dict() and 'px_width' in device_specs.dict():
            # If both attributes are present, calculate 'pixel_density' and add it to device_dict
            device_dict['pixel_density'] = device_dict['px_height'] * device_dict['px_width']
        else:
            # If either attribute is missing, handle the error or provide a default value
            print("Error: 'px_height' or 'px_width' is missing in device_specs.")

        # Filter out unwanted features
        device_dict = {key: value for key, value in device_dict.items() if key in selected_features}
        # Convert dictionary to DataFrame
        df = pd.DataFrame([device_dict])
        # Make a prediction using the trained model
        print(type(df))
        # Make a prediction using the trained model
        predicted_price = GB_model.predict(df)[0]

        # Convert the prediction result to a native Python data type
        predicted_price = int(predicted_price)  # Convert numpy.int64 to int

        # Return the predicted price
        return {'predicted_price': predicted_price}
    except Exception as e:
        return {'error': str(e)}


@app.post('/train_and_test')
async def train_and_test():
    # Load train dataset
    train_url = "train - train.csv"
    train_df = pd.read_csv(train_url)

    # Load test dataset
    test_url = "test - test.csv"
    test_df = pd.read_csv(test_url)

    # EDA(df=train_df)
    train_df=preprocess(train_df)
    test_df=preprocess(test_df)
    train_df=feature_engineering(train_df)
    test_df=feature_engineering(test_df)
    selected_features=select_features(train_df)
    select_features_train=train_df[selected_features]
    # Remove 'price_range' column from selected_features for test_df
    selected_features.remove('price_range')
    select_features_test = test_df[selected_features]
    filename = 'selected_features.txt'
    save_selected_features(select_features_train.columns, filename)

    GB=GradientBoostingclassifier(select_features_train)
    GB.train_test_split()
    GB.train_test_model()

    RF=RandomForestclassifier(select_features_train)
    RF.train_test_split()
    RF.train_test_model()


    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8001)
