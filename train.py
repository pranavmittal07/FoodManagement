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
import joblib  # To save the model

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

# Train the model
ensemble_model.fit(X, y)

# Save the model to a file
joblib.dump(ensemble_model, 'restaurant_operations_model.pkl')

print("Model training complete and saved to 'restaurant_operations_model.pkl'")
