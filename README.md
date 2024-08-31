# Module Overview

This module was developed as part of a group project in the 'Advanced Programming for Data Science' course at NOVA School of Business and Economics.
The module provides a comprehensive solution for analyzing and visualizing 
flight data. It includes the Flights class, designed for various 
tasks like plotting airports in countries, creating flight distance histograms,
and visualizing flight paths. The class manages data download, preprocessing,
and visualization with libraries like Pandas, Matplotlib, and Geopandas. It also incorporates OpenAI’s language models, which support the analysis with detailed descriptions and specifications.

Flights automates flight data retrieval, processes the data for 
enhanced utility (such as calculating distances and merging datasets), and 
provides methods for detailed visual analyses. These capabilities enable 
in-depth study of global flight operations, including 
distance analysis, and identifying popular airplane models.

# Requirements:

Before you begin, ensure you have the following prerequisites installed on your system:
- Python 3
- conda or Miniconda

These are essential for creating and managing the project environment using the provided `Group_20_Environment.yaml` file.

# How to Run the Project

1. Clone the Repository

2. Create and activate the Conda environment
  
  Use the provided Group_20_Environment.yaml file to create an environment with all          necessary dependencies:

        $ conda env create -f Group_20_Environment.yaml

  Wait for Conda to set up the environment. Once it's done, you can activate it. With the    environment active, all project dependencies are available for use.

3. Launch the Showcase_Notebook.ipynb notebook to view a brief showcase of the main functionalities of the module.

# Flights Class Details

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

# Remarks

Setting the PEP 8 compliance threshold of pylint to 8 is a practical compromise that balances code quality with developer efficiency.  It upholds Python's style guidelines while offering flexibility for cases where perfect adherence isn't feasible or required. This approach helps keep the code readable and maintainable without turning linting into a development bottleneck. It fosters a culture of quality while addressing the practical challenges of software development.

# Authors

- Davide Rebuzzini
- Leonardo Heinemann
- Markus Giesbrecht
- Michel Oeding-erdel

# License

This project is licensed under the GNU General Public License - see the LICENSE.md file for details

