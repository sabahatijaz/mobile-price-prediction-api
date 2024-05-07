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

def load_model(file_path):
    return joblib.load(file_path)

def predict_price(device_specs: Device, GB_model):
    selected_features = load_selected_features("selected_features.txt")
    selected_features.remove('price_range')
    device_dict = device_specs.dict()
    if 'px_height' in device_dict and 'px_width' in device_dict:
        device_dict['pixel_density'] = device_dict['px_height'] * device_dict['px_width']
    device_dict = {key: value for key, value in device_dict.items() if key in selected_features}
    df = pd.DataFrame([device_dict])
    predicted_price = GB_model.predict(df)[0]
    return int(predicted_price)


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
async def predict(device_specs: Device):
    try:
        GB_model = load_model('GB_trained_model.pkl')
        price = predict_price(device_specs, GB_model)
        return {'predicted_price': price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post('/train_and_test')
async def train_and_test():
    try:
        train_df = pd.read_csv("train - train.csv")
        test_df = pd.read_csv("test - test.csv")
        EDA(train_df)
        train_df = preprocess(train_df)
        test_df = preprocess(test_df)
        train_df = feature_engineering(train_df)
        test_df = feature_engineering(test_df)
        selected_features = select_features(train_df)
        select_features_train = train_df[selected_features]
        selected_features.remove('price_range')
        select_features_test = test_df[selected_features]
        filename = 'selected_features.txt'
        save_selected_features(select_features_train.columns, filename)
        GB = GradientBoostingClassifier(select_features_train)
        GB.train_test_split()
        GB.train_test_model()
        RF = RandomForestClassifier(select_features_train)
        RF.train_test_split()
        RF.train_test_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8001)
