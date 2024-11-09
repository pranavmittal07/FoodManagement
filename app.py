import streamlit as st
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# MongoDB connection URI
uri = "mongodb+srv://UserP:MGSdJSmOtRMJ33Er@cluster0.itxnkx2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)
db = client['hotelmanagement']
collection = db['users']

# List of order IDs to fetch data
order_ids = [
    "672e5f4ba2f29b795d446b09",
    "672e69b856987cf7cbade8fa",
    "672ef0eeeaf2673aa0966824",
    "672ef11deaf2673aa0966828",
    "672ef133eaf2673aa096682c"
]

# Fetch data and prepare the DataFrame
all_data = []
for order_id in order_ids:
    restaurant_data = collection.find_one({"_id": ObjectId(order_id)})
    if restaurant_data and 'dailyData' in restaurant_data:
        for record in restaurant_data['dailyData']:
            record['restaurant_id'] = order_id
            all_data.append(record)
df = pd.DataFrame(all_data)

# Derived features and encoding
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['day_of_week'] = df['date'].dt.day_name()
df['day_of_week_numeric'] = LabelEncoder().fit_transform(df['day_of_week'])

# Feature preprocessing
X = df[['restaurant_id', 'day_of_week_numeric', 'is_holiday', 'weather_score', 'event_indicator', 
        'flour_consumed', 'rice_consumed', 'pulses_consumed', 'vegetables_consumed', 'seating_capacity']]
y = df[['attendance', 'flour_prepared', 'rice_prepared', 'pulses_prepared', 'vegetables_prepared']]

categorical_features = ['weather_score', 'restaurant_id']
numeric_features = ['day_of_week_numeric', 'is_holiday', 'event_indicator', 'flour_consumed', 
                    'rice_consumed', 'pulses_consumed', 'vegetables_consumed', 'seating_capacity']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

# Models and pipeline
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
ensemble_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(VotingRegressor([('xgb', xgb), ('rf', rf)])))
])

# Model training
ensemble_model.fit(X, y)

# Streamlit UI

st.title("Restaurant Operations Prediction")

# Input fields
restaurant_id = st.selectbox("Select Restaurant ID", df['restaurant_id'].unique())
day_of_week = st.selectbox("Select Day of Week", df['day_of_week'].unique())
is_holiday = st.selectbox("Is it a Holiday?", [1, 0])
weather_score = st.selectbox("Weather Score", ["sunny", "rainy", "snowy", "overcast"])
event_indicator = st.selectbox("Event Indicator", [1, 0])
seating_capacity = st.number_input("Seating Capacity", min_value=1)

# Simulate consumption data based on seating capacity
flour_consumed = seating_capacity * np.random.uniform(0.2, 0.3)
rice_consumed = seating_capacity * np.random.uniform(0.1, 0.2)
pulses_consumed = seating_capacity * np.random.uniform(0.05, 0.1)
vegetables_consumed = seating_capacity * np.random.uniform(0.03, 0.07)

# Prepare input data for prediction
day_of_week_numeric = LabelEncoder().fit_transform([day_of_week])[0]

input_data = pd.DataFrame([{
    'restaurant_id': restaurant_id,
    'day_of_week_numeric': day_of_week_numeric,
    'is_holiday': is_holiday,
    'weather_score': weather_score,
    'event_indicator': event_indicator,
    'flour_consumed': flour_consumed,
    'rice_consumed': rice_consumed,
    'pulses_consumed': pulses_consumed,
    'vegetables_consumed': vegetables_consumed,
    'seating_capacity': seating_capacity
}])

# Make prediction
if st.button("Predict"):
    prediction = ensemble_model.predict(input_data)
    st.write("### Predicted Results")
    st.write(f"Predicted Attendance: {prediction[0][0]}")
    st.write(f"Predicted Flour Prepared: {prediction[0][1]}")
    st.write(f"Predicted Rice Prepared: {prediction[0][2]}")
    st.write(f"Predicted Pulses Prepared: {prediction[0][3]}")
    st.write(f"Predicted Vegetables Prepared: {prediction[0][4]}")

