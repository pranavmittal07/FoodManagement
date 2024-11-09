import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient
from bson import ObjectId
import streamlit as st

# MongoDB connection
uri = "mongodb+srv://UserP:MGSdJSmOtRMJ33Er@cluster0.itxnkx2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)
db = client['hotelmanagement']
collection = db['users']

# Function to fetch data for a given username
def fetch_hostel_data(username):
    hostel_data = collection.find_one({"name": username})  # Fetch data based on username
    if hostel_data and 'dailyData' in hostel_data:
        return pd.DataFrame(hostel_data['dailyData'])
    else:
        st.error(f"Error: No data found for username {username}")
        return None

# Streamlit UI
st.title("Hostel Data Analysis")

# Fetch available usernames from MongoDB
usernames = [user['name'] for user in collection.find({}, {"name": 1, "_id": 0})]

# User selection for username
selected_username = st.selectbox("Select Username", usernames)

# Fetch data for the selected username
df = fetch_hostel_data(selected_username)

if df is not None:
    # Convert 'date' column to datetime if it's not already in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Add columns for day of the week and month
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month_name()

    # Convert relevant columns to numeric (if not already)
    df[['attendance', 'flour_consumed', 'rice_consumed', 'pulses_consumed', 'vegetables_consumed', 'seating_capacity',
        'flour_prepared', 'rice_prepared', 'pulses_prepared', 'vegetables_prepared']] = df[['attendance', 
        'flour_consumed', 'rice_consumed', 'pulses_consumed', 'vegetables_consumed', 'seating_capacity',
        'flour_prepared', 'rice_prepared', 'pulses_prepared', 'vegetables_prepared']].apply(pd.to_numeric, errors='coerce')

    # Calculate food waste for each item
    df['food_waste_flour'] = df['flour_prepared'] - df['flour_consumed']
    df['food_waste_rice'] = df['rice_prepared'] - df['rice_consumed']
    df['food_waste_pulses'] = df['pulses_prepared'] - df['pulses_consumed']
    df['food_waste_vegetables'] = df['vegetables_prepared'] - df['vegetables_consumed']

    # Total food waste
    df['food_waste'] = df[['food_waste_flour', 'food_waste_rice', 'food_waste_pulses', 'food_waste_vegetables']].sum(axis=1)

    # 1. *Average Attendance Per Day of the Week*
    avg_customers_per_day = df.groupby('day_of_week')['attendance'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    st.subheader('1. Average Number of Customers for Each Day of the Week')
    st.bar_chart(avg_customers_per_day)

    # 2. *Weekly Food Waste Trends*
    df['week'] = df['date'].dt.isocalendar().week
    weekly_food_waste = df.groupby('week')['food_waste'].sum()

    st.subheader('2. Weekly Food Waste for the Year')
    st.line_chart(weekly_food_waste)

    # 3. *Monthly Attendance Trends*
    monthly_attendance = df.groupby('month')['attendance'].sum().reindex(
        ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])

    st.subheader('3. Monthly Attendance Trend')
    st.bar_chart(monthly_attendance)

    # 5. *Average Food Consumption Per Customer*
    food_consumption_per_customer = df[['flour_consumed', 'rice_consumed', 'pulses_consumed', 'vegetables_consumed']].div(df['attendance'], axis=0).mean()

    st.subheader('4. Average Food Consumption per Customer')
    st.bar_chart(food_consumption_per_customer)

    # 6. *Derived Features: Day of the Year and Days Since Last Holiday*
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_holiday'] = df['date'].dt.isocalendar().week.isin([1, 12, 13, 17])  # Example: mark certain weeks as holidays
    df['days_since_last_holiday'] = (df['is_holiday'].cumsum().shift(1).fillna(0))

    # 7. *Impact of Weather on Attendance*
    if 'weather_score' in df.columns:
        st.subheader('5. Attendance vs Weather Score')
        
        # Create the box plot using seaborn and matplotlib
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='weather_score', y='attendance')
        st.pyplot()

    # 8. *Impact of Events on Attendance*
    if 'event_indicator' in df.columns:
        st.subheader('6. Attendance vs Event Indicator')
        
        # Create the box plot using seaborn and matplotlib
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='event_indicator', y='attendance')
        st.pyplot()

else:
    st.write("No data available to analyze.")
