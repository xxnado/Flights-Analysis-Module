"""
Flight Data Analysis

This module provides a comprehensive solution for analyzing and visualizing 
flight data. It includes the Flights class, designed for various 
tasks like plotting airports in countries, creating flight distance histograms,
and visualizing flight paths. The class manages data download, preprocessing,
and visualization with libraries like Pandas, Matplotlib, and Geopandas.

Flights automates flight data retrieval, processes the data for 
enhanced utility (such as calculating distances and merging datasets), and 
provides methods for detailed visual analyses. These capabilities enable 
in-depth study of global flight operations, including 
distance analysis, and identifying popular airplane models.

# Usage:
from flights_class_20 import Flights

# Create an analysis tool instance
flights_instance = Flights()

# Use class methods for different analyses

Requirements:
- Internet connection for data download.
- Pandas for data manipulation.
- Matplotlib for mapping.
- and others, included in the yaml file.
"""

import io
import math
import os
import zipfile


import geopandas as gpd
import pandas as pd
import requests
from IPython.display import Markdown
from matplotlib import pyplot as plt
from langchain_openai import ChatOpenAI


class Flights:
    """
    The Flights class offers methods for downloading and processing flight data, 
    including information about airports, airlines, routes, and airplanes. It also
    provides functionality for plotting various aspects of the flight data, such as 
    flight routes, airport locations, and the most used airplane models.

    Attributes:
    -----------
    airports: pandas DataFrame containing information about various airports.
    airlines: pandas DataFrame containing information about various airlines.
    routes: pandas DataFrame containing information about different flight routes.
    airplanes: pandas DataFrame containing information about different airplane models.
    geo_dataframe: GeoPandas DataFrame containing geographical data of countries.

    Methods:
    --------
    plot_airports(country: str) -> plt.Figure:
        Plots the location of airports in a given country on a map.
    haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        Calculates the distance between two points on the Earth's surface.
    distance_analysis(show_plot: bool=True) -> None:
        Analyzes and plots the distribution of distances covered by all flight routes.
    plot_flight_routes(airport_code: str, internal: bool = False) -> plt.Figure:
        Plots the number of flights departing from a given airport. If 'internal' is True, 
        only domestic flights are considered.
    plot_most_used_airplane_models(countries: str | list[str] =
        None, number: int = 10) -> plt.Figure:
        Plots the top N most used airplane models based on the number of routes,
        optionally filtered by country.
    aircrafts() -> None:
        Prints a list of unique aircraft names available in the dataset.
    aircraft_info(aircraft_name: str) -> Markdown:
        Displays detailed information about a specific aircraft in markdown format.
    airports_list() -> None:
        Prints a list of unique airport names available in the dataset.
    airport_info(airport_name: str) -> Markdown:
        Displays detailed information about a specific airport in markdown format.
    plot_flights_by_country(country_name: str, internal: bool =
        False, distance_cutoff: int = 1000) -> plt.Figure:
        Visualizes flight route distributions from a specified country,
        distinguishing between domestic and international,
        and short-haul versus long-haul flights.

    Raises:
    -------
    ValueError: If a specified country or airport code is not found in the dataset.

    Examples:
    ---------
    # Create an instance of Flights
    flights_instance = Flights()

    # Plot flight routes from Frankfurt Airport, including only internal flights
    flights_instance.plot_flight_routes('FRA', internal=True)

    # Plot locations of airports in Germany
    flights_instance.plot_airports('Germany')

    # Plot the top 10 most used airplane models in Germany
    flights_instance.plot_most_used_airplane_models('Germany')

    # Analyze and plot the distribution of flight distances
    flights_instance.distance_analysis()

    # Visualize flights departing from Germany, including only internal flights
    flights_instance.plot_flights_by_country('Germany', internal=True)
    """
    def __init__(self):
        self.airports = None
        self.airlines = None
        self.routes = None
        self.airplanes = None
        self.geo_dataframe = None

        absolute_path = os.path.dirname(__file__)
        relative_path = "/../downloads"
        full_path = absolute_path + relative_path

        if not os.path.exists(full_path):
            os.mkdir(full_path)

        # url of the flight data datasets
        url_1 = (
    "https://gitlab.com/adpro1/adpro2024/-/raw/main/"
    "Files/flight_data.zip?inline=false"
)

        # download the zip file and extract the csv files
        response = requests.get(url_1, timeout=20)
        with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
            # Extrahieren jeder CSV-Datei aus dem ZIP und Laden in separate DataFrames
            for zipinfo in thezip.infolist():
                if zipinfo.filename.endswith('.csv'):
                    thezip.extract(zipinfo, full_path)
                    file_path = os.path.join(full_path, zipinfo.filename)
                    if 'airport' in zipinfo.filename:
                        self.airports = pd.read_csv(file_path)
                        self.airports.drop(['Source', 'Type'], axis=1, inplace=True)
                    elif 'airlines' in zipinfo.filename:
                        self.airlines = pd.read_csv(file_path)
                        self.airlines.drop(['Alias', 'Callsign'], axis=1, inplace=True)
                    elif 'routes' in zipinfo.filename:
                        self.routes = pd.read_csv(file_path)
                        self.routes.drop('Codeshare', axis=1, inplace=True)
                    elif 'airplanes' in zipinfo.filename:
                        self.airplanes = pd.read_csv(file_path)

        # geographical data of countries
        url_2 = (
    "https://d2ad6b4ur7yvpq.cloudfront.net/"
    "naturalearth-3.3.0/ne_50m_admin_0_countries.geojson"
)
        self.geo_dataframe = gpd.read_file(url_2)

    def plot_airports(self, country: str) -> plt.Figure:
        """
        Plots the geographical locations of airports within a specified country on a map. 
        This method checks if the country exists in the dataset,
        filters the airports in that country, 
        and then plots them on a world map with the country highlighted.

        Parameters:
        -----------
        country: str
            The name of the country for which to plot the airports.

        Returns:
        --------
        plt.Figure 
            Returns a matplotlib Figure object containing the plot.

        Raises:
        -------
        TypeError:
            If the 'country' argument is not a string.
        ValueError:
            If the specified country is not found in the dataset.

        Examples:
        ---------
        # Plot airports in Germany
        flights_instance.plot_airports('Germany')
        """

        # Type checking
        if not isinstance(country, str):
            raise TypeError(
                "The 'country' argument must be str, but is "
                f"{type(country).__name__}."
            )
        # Check if country data is available
        if country not in self.airports["Country"].values:
            raise ValueError(f"Country '{country}' not available in the dataset.")

        # Filter the dataset for the specified country
        country_airports = self.airports[self.airports['Country'] == country]

        # Make sure the country's map and the airports' points are using the same CRS
        country_map = self.geo_dataframe[self.geo_dataframe.name == country]
        country_map = country_map.to_crs(epsg=4326)  # WGS84 Lat/Long

        # Plot the world map focused on the country
        fig, ax = plt.subplots()
        country_map.plot(ax=ax, color='white', edgecolor='black')

        # Plot the airports on the map
        gdf = gpd.GeoDataFrame(
            country_airports,
            geometry=gpd.points_from_xy(country_airports.Longitude, country_airports.Latitude)
        )
        gdf = gdf.set_crs(epsg=4326)
        gdf.plot(ax=ax, color='red', marker='o', markersize=5)

        # Set axis limits to the bounds of the country
        minx, miny, maxx, maxy = country_map.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        plt.title(f'Airports in {country}')

        return fig

    def haversine(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """
        Calculates the great-circle distance between two points
        on the Earth's surface given their longitude and latitude.
        This method uses the Haversine formula to compute the distance.

        Parameters:
        -----------
        lon1: float
            Longitude of the first point.
        lat1: float
            Latitude of the first point.
        lon2: float
            Longitude of the second point.
        lat2: float
            Latitude of the second point.

        Returns:
        --------
        float
            The distance in kilometers between the two points.

        Raises:
        -------
        TypeError:
            If not all the 'Latitude and Longitude' arguments are floats.

        Examples:
        ---------
        # Calculate the distance between London (51.5074° N, 0.1278° W)
        and New York (40.7128° N, 74.0060° W)
        distance = flights_instance.haversine(-0.1278, 51.5074, -74.0060, 40.7128)
        print(distance)
        """

        # Type checking for each variable individually
        if not all(isinstance(var, float) for var in [lon1, lat1, lon2, lat2]):
            raise TypeError(
                "The Latitude and Longitude arguments must be float, but at least one is not."
            )
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

        # Haversine-Formel
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        radius = 6371  # radius of the earth
        return c * radius

    def distance_analysis(self, show_plot: bool=True) -> None:
        """
        Analyzes and plots the distribution of distances
        covered by all flight routes. 
        This method uses the Haversine formula to calculate
        the distance between source and destination airports 
        for each route and adds a new column 'Distance' to the
        routes DataFrame to store these distances. 
        Optionally, it can also plot a histogram of
        these distances based on the show_plot parameter.

        Parameters:
        -----------
        show_plot : bool, optional (default is True)
            If True, plot the distribution of flight distances. Show_plt=False can be used
            to only create the column 'Distance' in the routes DataFrame but not plot the 
            histogram.

        Returns:
        --------
        None

        Examples:
        ---------
        # Analyze and plot the distribution of flight distances
        flights_instance.distance_analysis2()

        # Analyze the distribution of flight distances without plotting
        flights_instance.distance_analysis2(show_plot=False)

        Note:
        -----
        The return is None but the methods plots a histogram of
        the distribution of flight distances and adds a new column
        'Distance' to the routes DataFrame.
        """

        # Initialize the 'Distance' column in the routes DataFrame
        self.routes['Distance'] = None

        for index, route in self.routes.iterrows():
            source_airports = self.airports[self.airports['IATA'] == route['Source airport']]
            dest_airports = self.airports[self.airports['IATA'] == route['Destination airport']]

            if not source_airports.empty and not dest_airports.empty:
                source_airport = source_airports.iloc[0]
                dest_airport = dest_airports.iloc[0]

                distance = self.haversine(source_airport['Longitude'], source_airport['Latitude'],
                                          dest_airport['Longitude'], dest_airport['Latitude'])
                # Assign the calculated distance to the 'Distance' column for the current row
                self.routes.at[index, 'Distance'] = distance

        # Plot the histogram only if show_plot is True
        if show_plot:
            plt.hist(self.routes['Distance'].dropna(), bins=30, edgecolor='black')
            plt.title('Distribution of Flight Distances')
            plt.xlabel('Distance (km)')
            plt.ylabel('Number of Flights')
            plt.show()

    def plot_flight_routes(self, airport_code: str, internal: bool = False) -> plt.Figure:
        """
        Plots the number of flights departing from a specific airport. Optionally, 
        the method can filter the flights to show only domestic (internal) routes.

        Parameters:
        -----------
        airport_code: str
            The IATA code of the airport to analyze.
        internal: bool, optional (default is False)
            If set to True, only domestic flights from the specified airport are considered.

        Returns:
        --------
        plt.Figure
            Returns a matplotlib Figure object containing the plot.

        Raises:
        -------
        TypeError:
            If the 'airport_code' argument is not a string,
            or if the 'internal' argument is not a boolean.
        ValueError:
            If the given airport code is not found in the dataset.

        Examples:
        ---------
        # Plot all flights from Frankfurt Airport
        flights_instance.plot_flight_routes('FRA')

        # Plot only domestic flights from Frankfurt Airport
        flights_instance.plot_flight_routes('FRA', internal=True)
        """
        # Type checking
        if not isinstance(airport_code, str):
            raise TypeError(
                f"The 'airport_code' argument has to be of type str, but is"
                f"{type(airport_code).__name__}."
            )
        if not isinstance(internal, bool):
            raise TypeError(
                f"The 'internal' argument must be boolean, but is"
                f"{type(internal).__name__}."
            )

        # Checking for airport
        if airport_code not in self.routes["Source airport"].values:
            raise ValueError(f"Airport '{airport_code}' is not available in the DataFrame.")

        start_airport_row = self.airports[self.airports['IATA'] == airport_code].iloc[0]
        start_country = start_airport_row['Country']
        flights_from_airport = self.routes[self.routes['Source airport'] == airport_code]

        if internal:
            # find all airports in the same country as the start airport
            airports_in_country = self.airports[self.airports['Country'] ==
                                                start_country]['IATA'].unique()
            # filter the flights to only include those landing within the same country
            internal_flights = flights_from_airport[flights_from_airport
                                            ['Destination airport'].isin(airports_in_country)]
            flight_counts = internal_flights['Destination airport'].value_counts()
        else:
            # consider all flights departing from the specified airport
            flight_counts = flights_from_airport['Destination airport'].value_counts()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting the bar chart using the Axes object 'ax'
        flight_counts.plot(kind='bar', ax=ax)

        # Setting the title and labels using the Axes object 'ax'
        ax.set_title(f"Flights from {airport_code}" + (" (Internal Only)" if internal else ""))
        ax.set_xlabel('Destination Airport')
        ax.set_ylabel('Number of Flights')

        # Get the current xtick locations and labels from 'ax'
        xticks_locs, xticks_labels = ax.get_xticks(), ax.get_xticklabels()

        # Only keep every 10th xtick
        ax.set_xticks(xticks_locs[::10])
        ax.set_xticklabels(xticks_labels[::10], rotation=45)

        # Use 'tight_layout' to automatically adjust subplot params
        fig.tight_layout()

        return fig

    def plot_most_used_airplane_models(self, countries: str | list[str] =
                                       None, number: int = 10) -> plt.Figure:
        """
        Visualizes the most frequently used airplane models based on the number of routes. 
        This method can be filtered by country or countries to show data specific to those regions. 
        If no country is specified, it considers the entire dataset.

        Parameters:
        -----------
        countries: Union[str, List[str]], optional
            A single country or a list of countries for which to filter the data. 
            If None, the data is not filtered by country.
        number: int, optional (default is 10)
            The number of top airplane models to display.

        Returns:
        --------
        plt.Figure
            Returns a matplotlib Figure object containing the plot.

        Raises:
        -------
        TypeError:
            If the 'number' argument is not an integer, or if the
            'countries' argument is not a string or a list of strings.
        ValueError:
            If any country specified in the countries is not found in the dataset.

        Examples:
        ---------
        # Plot the top 10 most used airplane models globally
        flights_instance.plot_most_used_airplane_models()

        # Plot the top 5 most used airplane models in Germany and France
        flights_instance.plot_most_used_airplane_models(countries=['Germany', 'France'], number=5)
        """

        # Type and value checking
        if not isinstance(number, int):
            raise TypeError(
                f"The 'number' argument must be int, but got"
                f"{type(number).__name__}."
            )

        if not (
            countries is None
            or countries == []
            or isinstance(countries, str)
            or (
                isinstance(countries, list)
                and all(isinstance(item, str) for item in countries)
            )
        ):
            raise TypeError(
                "The 'countries' argument must be None, an empty list, str, or list[str]."
            )

        if isinstance(countries, str):
            countries = [countries]

        countries_not_found = []
        for country in countries:
            if country not in list(self.airports["Country"]):
                countries_not_found.append(country)

        if len(countries_not_found) > 0:
            raise ValueError(
                f"Countries '{countries_not_found}' are not in  DataFrame."
            )

        # Check if the 'countries' argument is a string and convert it to a list
        if countries:
            if isinstance(countries, str):
                countries = [countries]
            airports_in_countries = self.airports[self.airports
                                                  ['Country'].isin(countries)]['IATA'].unique()
            filtered_routes = self.routes[(self.routes
                            ['Source airport'].isin(airports_in_countries))
                            | (self.routes['Destination airport'].isin(airports_in_countries))]
        else:
            filtered_routes = self.routes

        model_counts = (
    filtered_routes['Equipment'].str.split(expand=True)
    .stack()
    .value_counts()
    .head(number)
)

        fig, ax = plt.subplots(figsize=(10, 6))
        model_counts.plot(kind='bar', ax=ax)  # Ensure to plot on the ax object
        ax.set_title('Top N Most Used Airplane Models' +
                     (' in ' + ', '.join(countries) if countries else ''))
        ax.set_xlabel('Airplane Model')
        ax.set_ylabel('Number of Routes')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        fig.tight_layout()

        return fig

    def aircrafts(self) -> None:
        """
        Prints a list of unique aircraft names available in the dataset.

        This method checks if the aircraft data is loaded and then prints the names of
        all unique aircraft models. If no aircraft data is available, it prints an 
        appropriate message indicating the absence of data.

        Returns:
        --------
        None

        Examples:
        ---------
        # Print the list of aircraft models
        flights_instance.aircrafts()
        """

        # Check if the list of airplanes is available
        if self.airplanes is not None:
            # Print the list of aircraft models
            for aircraft in self.airplanes['Name'].unique():
                print(aircraft)
        else:
            print("No aircraft data available.")

    def aircraft_info(self, aircraft_name: str) -> Markdown:
        """
        Displays detailed information about a specific aircraft in markdown format.

        Parameters:
        -----------
        aircraft_name : str
            The name of the aircraft model for which information is requested.
        
        Returns:
        --------
        str
            A markdown table displaying the specifications
            and details of the requested aircraft model.
            
        Raises:
        -------
        Exception
            If the specified aircraft is not found in the dataset
            or if the OpenAI API key is not found.

        Examples:
        ---------
        # Display detailed information about the Ilyushin IL96
        flights_instance.aircraft_info('Ilyushin IL96')
        """

        # Check if the aircraft is in the list
        if aircraft_name not in self.airplanes['Name'].unique():
            raise Exception(
    f"Aircraft '{aircraft_name}' could not be found. "
    "Please use the method aircrafts to find available aircraft models."
)

        # Retrieve the API key from an environment variable
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise Exception(
    "OpenAI API key not found. Please set the "
    "OPENAI_API_KEY environment variable."
)

        llm = ChatOpenAI(temperature=0.3, openai_api_key=openai_api_key)
        result = llm.invoke(
    f'Print out a table of specifications about the aircraft {aircraft_name}'
)

        return Markdown(result.content)

    def airports_list(self) -> None:
        """
        Prints a list of unique airport names available in the dataset.

        This method checks if the airport data is loaded and then prints the names of
        all unique airports. If no airport data is available, it prints an
        appropriate message indicating the absence of data.

        Returns:
        --------
        None

        Examples:
        ---------
        # Print the list of airports
        flights_instance.airports_list()
        """

        # Check if the list of airports is available
        if self.airports is not None:
            # Print the list of airports
            for airports in self.airports['Name'].unique():
                print(airports)
        else:
            print("No airport data available.")

    def airport_info(self, airport_name: str) -> Markdown:
        """
        Displays detailed information about a specific airport in markdown format.

        Parameters:
        -----------
        airport_name : str
            The name of the airport for which information is requested.

        Raises:
        -------
        Exception
            If the specified airport is not found in the dataset
            or if the OpenAI API key is not found. 

        Shows:
        ------
        A markdown table displaying the specifications and details of the
        requested airport.

        Returns:
        --------
        None
        """

        # Check if the airport is in the list
        if airport_name not in self.airports['Name'].unique():
            raise Exception(
    f"Airport '{airport_name}' could not be found. "
    "Please use the method airports_list to find available airports."
)

        # Retrieve the API key from an environment variable
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise Exception(
    "OpenAI API key not found. Please set "
    "the OPENAI_API_KEY environment variable."
)

        llm = ChatOpenAI(temperature=0.3, openai_api_key=openai_api_key)
        result = llm.invoke(f'Print out a table of specifications about the airport {airport_name}')

        return Markdown(result.content)

    def plot_flights_by_country(
    self,
    country_name: str,
    internal: bool = False,
    distance_cutoff: int = 1000
) -> plt.Figure:
        """
        Visualizes flight route distributions from a specified country, distinguishing between
        domestic and international, and short-haul versus long-haul flights. Calculates and
        annotates the potential emissions reduction from replacing short-haul flights with
        rail services, based on a distance cutoff.

        Parameters:
        -----------
        country_name : str
            Name of the departure country.
        internal : bool, optional
            Consider only domestic flights if True.
        distance_cutoff : int, optional
            Distance in kilometers to define short-haul flights.

        Returns:
        --------
        plt.Figure
            Returns a matplotlib Figure object containing the histogram.
        
        Raises:
        -------
        TypeError:
            If the 'country_name' argument is not a string,
            or if the 'internal' argument is not a boolean,
            or if the 'distance_cutoff' argument is not an int.
        ValueError:
            If the specified country is not found in the dataset.

        Shows:
        ------
        A bar chart visualizing the number of short-haul and long-haul flights,
        with annotations for unique routes and potential emissions savings.

        Examples:
        ---------
        # Visualize flights departing from Germany, including only internal flights
        flights_instance.plot_flights_by_country('Germany', internal=True)
        """

        # Type checking
        if not isinstance(country_name, str):
            raise TypeError(
                f"The 'country_name' argument must be a str, but is "
                f"{type(country_name).__name__}."
            )
        if country_name not in self.airports['Country'].unique():
            raise ValueError(
                f"The country '{country_name}' is not available in the dataset."
            )
        if not isinstance(internal, bool):
            raise TypeError(
                f"The 'internal' argument must be of type bool, but is "
                f"{type(internal).__name__}."
            )
        if not isinstance(distance_cutoff, int):
            raise TypeError(
                f"The 'distance_cutoff' argument must be a int, but is "
                f"{type(distance_cutoff).__name__}."
            )

        self.distance_analysis(show_plot=False)

        airports_in_country = self.airports[
            self.airports['Country'] == country_name]['IATA'].unique()
        flights_from_country = self.routes[
            self.routes['Source airport'].isin(airports_in_country)]

        if internal:
            flights_from_country = flights_from_country[
                flights_from_country['Destination airport'].isin(airports_in_country)]

        short_haul_counts = (
            flights_from_country[flights_from_country['Distance'] < distance_cutoff]
            ['Destination airport']
            .value_counts()
)

        long_haul_counts = (
            flights_from_country[flights_from_country['Distance'] >= distance_cutoff]
            ['Destination airport']
            .value_counts()
)

        short_haul_flights = flights_from_country[
            flights_from_country['Distance'] < distance_cutoff]
        short_haul_flights['Route'] = short_haul_flights.apply(
            lambda x: frozenset([x['Source airport'], x['Destination airport']]), axis=1)
        unique_short_haul_routes = short_haul_flights['Route'].unique()
        short_haul_flights['route_id'] = short_haul_flights.apply(
            lambda row: '_'.join(sorted([row['Source airport'], row['Destination airport']])), axis=1)

        unique_routes_distances = short_haul_flights.groupby('route_id')['Distance'].first()
        unique_short_haul_distances = unique_routes_distances.sum()

        figure, ax = plt.subplots(figsize=(20, 6))

        if not short_haul_counts.empty:
            short_haul_counts.plot(kind='bar', color='grey', position=0, width=0.4, label='Short-haul', ax=ax)
        else:
            print("No short-haul flights to plot.")

        if not long_haul_counts.empty:
            long_haul_counts.plot(kind='bar', color='greenyellow', position=1, width=0.4, label='Long-haul', ax=ax)
        else:
            print("No long-haul flights to plot.")

        ax.set_title(f"Flights from {country_name}" + (" (Internal Only)" if internal else ""))
        ax.set_xlabel('Destination Airport')
        ax.set_ylabel('Number of Flights')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.legend()

        emissions_factor = 90  # grams of CO2 per kilometer per passenger
        total_short_haul_emissions = unique_short_haul_distances * emissions_factor
        emissions_reduction = total_short_haul_emissions * 0.86
        emissions_reduction_tonnes = emissions_reduction / 1e6

        annotation_text = (
            f"Unique short-haul routes: {len(unique_short_haul_routes)}\n"
            f"Total short-haul distance: {unique_short_haul_distances:.2f} km\n"
            f"Potential emissions reduction: {emissions_reduction_tonnes:.2f} tonnes CO2 equivalent"
        )
        ax.annotate(annotation_text, xy=(0.05, 0.75), xycoords='axes fraction', fontsize=15,
                    bbox=dict(boxstyle="round", fc="w"))

        figure.tight_layout()  # Use fig.tight_layout() instead of plt.tight_layout()

        return figure


Flights()
