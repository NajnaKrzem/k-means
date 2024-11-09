import unittest
from uc import update_centroids
from ua import update_assignments
import numpy as np

class TestKMeansFunctions(unittest.TestCase):                                               

    def test_update_assignments(self):                                                      
        data = np.array([[1, 2], [2, 3], [3, 4]])
        centroids = np.array([[1, 2], [4, 5]])
        assignments = update_assignments(data, centroids)
        expected_assignments = [0, 0, 1]  
        self.assertEqual(assignments, expected_assignments)

    def test_update_centroids(self):
        data = np.array([[1, 2], [2, 3], [3, 4]])                                           
        assignments = [0, 0, 1]  
        new_centroids = update_centroids(data, 2, assignments)                              
        expected_centroids = np.array([[1.5, 2.5], [3, 4]])  
        np.testing.assert_array_almost_equal(new_centroids, expected_centroids)

if __name__ == '__main__':
    unittest.main() 

'''if __name__ == '__main__':
    unittest.main(argv=[''], exit=False) '''
