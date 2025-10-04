import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import mlflow
import mlflow.sklearn

#Load dataset
df = pd.read_csv(r"C:\Users\A S U S\Downloads\amazon_delivery - amazon_delivery.csv")

#Handle missing values
df.fillna(0, inplace=True)
df.drop_duplicates(inplace=True)

#Convert categorical columns to string
cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
for col in cols:
    df[col] = df[col].astype(str)

#Label encoding
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

#Convert date/time safely
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors="coerce", format="mixed")
df['Order_Time'] = pd.to_datetime(df['Order_Time'], errors="coerce", format="mixed")
df['Pickup_Time'] = pd.to_datetime(df['Pickup_Time'], errors="coerce", format="mixed")

#Drop rows with invalid dates
df.dropna(subset=['Order_Date', 'Order_Time', 'Pickup_Time'], inplace=True)

#Feature engineering
df['order_hour'] = df['Order_Time'].dt.hour
df['order_dayofweek'] = df['Order_Date'].dt.dayofweek
df['pickup_delay_mins'] = (df['Pickup_Time'] - df['Order_Time']).dt.total_seconds() / 60.0

#Drop unused columns
df.drop(['Order_ID', 'Order_Date', 'Order_Time', 'Pickup_Time',
         'Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude'],
        axis=1, inplace=True)

#Ensure all int columns are float (avoid MLflow schema warning)
for col in df.select_dtypes(include=['int']).columns:
    df[col] = df[col].astype(float)

#Split features and target
x = df.drop("Delivery_Time", axis=1)
y = df["Delivery_Time"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Evaluation function
def eval_metrics(actual, pred):
    rmse = root_mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

#Load saved models
models = {
    "Linear_Regression": "Linear_Regression.pkl",
    "Random_Forest": "Random_Forest.pkl",
    "Gradient_Boost": "Gradient_Boost.pkl"
}

#Set MLflow experiment
mlflow.set_experiment("Amazon_Delivery_time_prediction")

best_model = None
best_score = -np.inf  

#Evaluate and log models
for model_name, model_file in models.items():
    model = joblib.load(model_file)

    #Predict on test data
    predictions = model.predict(x_test)

    #Metrics
    rmse, mae, r2 = eval_metrics(y_test, predictions)

    #Log to MLflow
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        #Add input example for clean logging
        input_example = x_test.iloc[:1]
        mlflow.sklearn.log_model(model, name=model_name, input_example=input_example)

    #Track best model
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = model_name

    print(f"{model_name} -> RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

#Save the best model for deployment
if best_model is not None:
    joblib.dump(best_model, "Best_Model.pkl")
    print(f"\nBest Model Saved: {best_name} with R2={best_score:.4f}")

print("Done")
