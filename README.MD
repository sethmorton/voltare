# Electric Vehicle Charging Time and Cost Estimator

This Python script estimates the time and cost of charging an electric vehicle (specifically a Tesla Model Y) based on various factors such as the current state of charge, target state of charge, current temperature, and electricity prices.

## Dependencies

The script uses the following Python libraries:

- os
- requests
- psycopg2
- pandas
- numpy
- haversine
- dotenv
- scipy

## How It Works

The script first sets up the necessary parameters such as database connection details, OpenWeatherMap API key, battery specifications, charging parameters, time parameters, and location coordinates.

It then defines several functions to perform various tasks:

- `execute_query(query)`: Connects to a PostgreSQL database and executes a SQL query.
- `get_current_temperature(api_key, latitude, longitude)`: Fetches the current temperature from the OpenWeatherMap API.
- `temperature_charge_decline(temperature, BASE_CHARGE_RATE)`: Calculates the decline in charge rate due to temperature.
- `high_charge_curve(target_charge, CHARGE_RATE)`: Calculates the charge rate curve for high charge levels.
- `time_to_charge(rate_mi_hr, miles_per_kwh, battery_capacity, charge_to, charged)`: Calculates the time to charge the battery.
- `get_time_hours()`: Calculates the time to charge the battery in hours.
- `get_dist(site, loc)`: Calculates the haversine distance from a point to a data collection site.
- `get_nearest_collection_site(loc, loc_df)`: Finds the nearest data collection site to a given location.
- `get_hourly_medians(site_df)`: Calculates the median price for each hour.
- `get_kde(lat, lon, charge_time, range_min, range_max)`: Calculates the Kernel Density Estimation (KDE) and integrates over a given range.

Finally, it calculates the KDE and integrates over a given range to estimate the cost of charging the electric vehicle.

## Usage

To use this script, simply run it in a Python environment. Make sure to set the necessary environment variables for the database connection and OpenWeatherMap API key.

## Collaborators

Seth M, Aryan J, Max N



## DISCLAIMER

Connection details for the PostgreSQL database and OpenWeatherMap API key have been removed from the script for security reasons. The script will not run without these details.
