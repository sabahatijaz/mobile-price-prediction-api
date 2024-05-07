import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

class GradientBoostingclassifier:
    def __init__(self,df) -> None:
        self.df=df
        self.X_train=None
        self.X_test=None
        self.y_train=None
        self.y_test=None
        self.model=None


    def train_test_split(self):
        # Split the data into features (X) and target variable (y)
        X = self.df.drop(['price_range'], axis=1)  # Features
        y = self.df['price_range']  # Target variable

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    def train_test_model(self):
        # Model Training
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        # Model Testing
        # Predict on the testing set
        y_pred = self.model.predict(self.X_test)

        # Evaluate the model
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)

        # Classification Report
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

        # Confusion Matrix
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        joblib.dump(self.model, 'GB_trained_model.pkl')
        print("model saved")

    def test_on_unseen_data(self,data):
        y_pred = self.model.predict(data)
        return y_pred
    

    

    
