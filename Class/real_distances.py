import pandas as pd
import math

# Suppose 'airports' is your DataFrame. Make sure to load your DataFrame before using this function.
# For example:
# airports = pd.read_csv('path_to_your_airports_data.csv')

def real_distances(airport_name1, airport_name2, airports_df):
    # Radius of the Earth in km
    R = 6371.0
    
    # Lookup the latitude and longitude for the first airport
    airport1 = airports_df[airports_df['Name'] == airport_name1]
    lat1 = airport1.iloc[0]['Latitude']
    lon1 = airport1.iloc[0]['Longitude']
    
    # Lookup the latitude and longitude for the second airport
    airport2 = airports_df[airports_df['Name'] == airport_name2]
    lat2 = airport2.iloc[0]['Latitude']
    lon2 = airport2.iloc[0]['Longitude']
    
    # Convert latitude and longitude from degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance in kilometers
    distance = R * c
    return distance

