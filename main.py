import os
import requests
import psycopg2
import pandas as pd
import numpy as np
import haversine
from dotenv import load_dotenv
from scipy import integrate
from scipy.stats import gaussian_kde
from psycopg2 import Error
import os

# Database connection parameters
db_params = {
    "host": os.environ["DB_HOST"],
    "database": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "port": os.environ["DB_PORT"]
}

# OpenWeatherMap API key
api_key = os.environ["WEATHER_KEY"]

# Battery specs
battery_specs = {
    "capacity": 82,  # kWh
    "charging_rate_peak": 44,  # Miles per hour at peak
    "charging_rate_factor": 0.8,  # Charging rate drops by this factor between peak and 80% SOC
    "miles_per_kwh": 3.65  # Average miles per kWh for Model Y
}

# Charging parameters
charging_params = {
    "charge_to": 80,  # Targeted Percent charge of battery capacity
    "charged": 15  # Current Percent charge of battery capacity
}

time_params = {
    "start_time" : 15,
    "end_time" : 20
}

# Location coordinates
location = {
    "latitude": 32.959492,
    "longitude": -117.265244
}

# Base charge rate at ideal temperature
BASE_CHARGE_RATE = 44

# Function to connect to the PostgreSQL database and execute a query
def execute_query(query):
    try:
        # Connect to the PostgreSQL database
        connection = psycopg2.connect(**db_params)

        # Create a cursor object
        cursor = connection.cursor()

        # Execute the SQL query
        cursor.execute(query)

        # Fetch all the rows from the result of the query
        records = cursor.fetchall()

        return records

    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)

    finally:
        # Close the database connection
        if connection:
            cursor.close()
            connection.close()

# Function to get the current temperature from the OpenWeatherMap API
def get_current_temperature(api_key, latitude, longitude):
    base_url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": api_key,
        "units": "imperial",  # Use "metric" for Celsius
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        temperature = data["main"]["temp"]
        return temperature
    else:
        print(f"Error: Unable to fetch weather data. Status code: {response.status_code}")
        return None

# Function to calculate the charge rate decline due to temperature
def temperature_charge_decline(temperature, BASE_CHARGE_RATE):
    rate = BASE_CHARGE_RATE

    if temperature < -20:
        rate *= 0.10

    elif temperature < 65:
        drop = (75 - temperature) // 8
        rate = BASE_CHARGE_RATE * (1 - 0.05 * drop)

    return rate

# Function to calculate the charge rate curve for high charge levels
def high_charge_curve(target_charge, CHARGE_RATE):

    if target_charge <= 80:
        return None

    else:
        # 1.5 is the factor of the charge rate drop
        rate = CHARGE_RATE / 1.5
        return rate

# Function to calculate the time to charge the battery
def time_to_charge(rate_mi_hr, miles_per_kwh, battery_capacity, charge_to, charged):
    # Initialize variables
    upper_charge_capacity = 0  # Charge capacity over 80%
    time_hrs = 0  # Amount of time it will take to charge
    charged_time_hrs = 0  # To subtract the time for the percent it has already been charged

    charged_capacity = battery_capacity * (charged / 100)  # How much capacity charged
    charge_capacity = battery_capacity * (charge_to / 100)  # How much target to charge

    if charge_to > 80:  # To charge above 80
        upper_charge_capacity = ((charge_to - 80)/100) * charge_capacity  # Capacity above 80% to charge
        battery_rate_mi_hr = high_charge_curve(charge_to, rate_mi_hr)  # Lowering charge rate w/ function
        rate_kwh_hr = battery_rate_mi_hr / miles_per_kwh  # converting mi/hr to kwh/hr
        time_hrs = upper_charge_capacity / rate_kwh_hr  # capacity over charge rate = time

    charge_capacity -= upper_charge_capacity  # subtract rest of the capacity
    rate_kwh_hr = rate_mi_hr / miles_per_kwh  # again for the rest of capacity
    charged_time_hrs += charged_capacity / rate_kwh_hr  # to subtract
    time_hrs += charge_capacity / rate_kwh_hr
    time_hrs -= charged_time_hrs  # take out the time already charged

    return time_hrs

# Function to calculate the time to charge the battery in hours
def get_time_hours():
    temperature = get_current_temperature(api_key, location["latitude"], location["longitude"])

    if temperature is not None:
        rate_mi_hr = temperature_charge_decline(temperature, BASE_CHARGE_RATE)
        time_hrs = time_to_charge(rate_mi_hr, battery_specs["miles_per_kwh"], battery_specs["capacity"], charging_params["charge_to"], charging_params["charged"])
        return time_hrs

# Function to calculate the haversine distance from a point to a data collection site
def get_dist(site, loc):
    collection_site_loc = (site['latitude'], site['longitude'])
    haversine_dist = haversine.haversine(loc, collection_site_loc)
    return {'title': site['title'], 'distance': haversine_dist}

# Function to find the nearest data collection site to a given location
def get_nearest_collection_site(loc, loc_df):
    return min([get_dist(site, loc) for index, site in loc_df.iterrows()], key=lambda x: x['distance'])['title']

# Function to calculate the median price for each hour
def get_hourly_medians(site_df):
    hourly_medians = {}
    for hour in range(0, 25):
        hour_df = site_df[site_df['starting_hour'] == hour]
        hourly_medians[hour] = np.median(hour_df['real_time_price'])
    return hourly_medians

# Function to calculate the KDE and integrate over a given range
def get_kde(lat, lon, charge_time, range_min, range_max):
    # Format location as a tuple
    loc = (lat, lon)

    # Get all locations with latitude and longitude
    loc_df = pd.DataFrame(execute_query('SELECT DISTINCT title, latitude, longitude FROM electricity_prices;'),
                          columns=['title', 'latitude', 'longitude'])

    # Get the nearest collection site
    nearest_collection_site = get_nearest_collection_site(loc, loc_df)

    # Get data associated with the nearest collection site
    query = "SELECT title, starting_hour, real_time_price FROM electricity_prices WHERE title = '" + nearest_collection_site + "' LIMIT 1000;"
    nearest_collection_site_data = pd.DataFrame(execute_query(query), columns=['Title', 'starting_hour', 'real_time_price'])

    # Get the median price for each hour
    hourly_medians = get_hourly_medians(nearest_collection_site_data)

    # Get the list of starting hours (0 to 24) as an array
    time = np.array(list(hourly_medians.keys()))

    # Get the median prices as an array
    price = np.array(list(hourly_medians.values()))

    # Create a KDE model
    kde = gaussian_kde(price, bw_method=0.6)

    # Generate a range of x values for the KDE plot
    x = np.linspace(time.min(), time.max(), 1000)

    # Calculate the log density
    log_density = kde.evaluate(x)

    start_index = np.argmax(x >= range_min)
    end_index = np.argmax(x >= range_max)

    subset_x = x[start_index:end_index]
    subset_log_density = log_density[start_index:end_index]

    # Get the minimum index of x values in the kernel
    min_index = np.argmin(subset_log_density)

    # Get the x value at the minimum index
    min_x = subset_x[min_index]

    # Get the y value at the minimum index
    min_density = np.exp(subset_log_density[min_index])

    def integrand(x):
        return np.exp(kde.evaluate(x)[0])

    # Integrate between the minimum x value and the minimum x value + the amount of time it takes to charge
    result, _ = integrate.quad(integrand, min_x, min_x + charge_time)

    return result, min_x, min_x + charge_time

# Calculate the KDE and integrate over a given range
print(get_kde(location["latitude"], location["longitude"], get_time_hours(), time_params['start_time'], time_params['end_time']))
