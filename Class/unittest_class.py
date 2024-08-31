import unittest
from Class.real_distances import real_distances
from Class.flights_class_20 import Flights  # Assuming class_20_with_URL contains the Flights class


class TestDistanceCalculator(unittest.TestCase):

    def test_same_location(self):
        # Test distance from a location to itself
        self.assertAlmostEqual(real_distances('John F Kennedy International Airport', 'John F Kennedy International Airport', Flights().airports), 0, places=2)
 

    def test_different_continents(self):
        # Test distance between JFK Airport (New York) and LHR Airport (London)
        self.assertAlmostEqual(real_distances('John F Kennedy International Airport', 'Los Angeles International Airport', Flights().airports), 3974.196710879195, places=0)


    def test_within_continent(self):
        # Test distance between CDG Airport (Paris) and FRA Airport (Frankfurt)
        self.assertAlmostEqual(real_distances('London Heathrow Airport', 'Frankfurt am Main Airport', Flights().airports), 654.7595966719059, places=0)


if __name__ == '__main__':
    # Run the tests
    unittest.main()
