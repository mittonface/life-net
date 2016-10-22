import unittest
import numpy as np
from gol_generator import Life


class LifeTester(unittest.TestCase):


    def test_init_default(self):
        """
        Test that we can init a lifescape with default size
        """
        life = Life()
        self.assertEqual((25, 25), life.lifescape.shape)

    def test_init_custom_size(self):
        """
        Test that we can init a lifescape of a custom size
        """
        life = Life((534, 10))
        self.assertEqual((534, 10), life.lifescape.shape)

    def test_set_from_array(self):
        life = Life()

        arr = np.array([[0, 1, 1, 0, 0, 1],
                        [0, 1, 0, 0, 1, 1],
                        [1, 0, 0, 1, 1, 0],
                        [0, 1, 1, 0, 0, 1],
                        [0, 1, 0, 0, 1, 1],
                        [1, 0, 0, 1, 1, 0]])

        life.set_from_array(arr)

        self.assertEqual((6, 6), life.lifescape.shape)

    def test_get_neighbours_no_edge(self):
        """
        Test that we can get the correct neighbour lists on non edge cells
        """
        life = Life()

        expected = [(20, 19), (21, 19), (22, 19), (20, 20), (22, 20), (20, 21), (21, 21), (22, 21)]
        actual = life._get_cell_neighbours((21,20))
        self.assertEqual(sorted(expected), sorted(actual))
        self.assertEqual(8, len(actual))

        expected = [(4, 4), (5, 4), (6, 4), (4, 5), (6, 5), (4, 6), (5, 6), (6, 6)]
        actual = life._get_cell_neighbours((5,5))
        self.assertEqual(sorted(expected), sorted(actual))
        self.assertEqual(8, len(actual))

        expected = [(4, 4), (4, 4), (6, 4), (4, 5), (6, 5), (4, 6), (5, 6), (6, 6)]
        actual = life._get_cell_neighbours((5,5))
        self.assertNotEqual(sorted(expected), sorted(actual))
        self.assertEqual(8, len(actual))

    def test_get_neighbours_on_top(self):
        """
        Test that we can get the correct neighbour lists on non edge cells
        """
        life = Life()

        expected = [(0, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        actual = life._get_cell_neighbours((1, 0))
        self.assertEqual(sorted(expected), sorted(actual))
        self.assertEqual(5, len(actual))

    def test_get_neighbours_on_left(self):
        """
        Test that we can get the correct neighbour lists on non edge cells
        """
        life = Life()

        expected = [(0, 4), (0, 6), (1, 4), (1, 5), (1, 6)]
        actual = life._get_cell_neighbours((0, 5))
        self.assertEqual(sorted(expected), sorted(actual))
        self.assertEqual(5, len(actual))

    def test_get_neighbours_on_bottom(self):
        """
        Test that we can get the correct neighbour lists on non edge cells
        """
        life = Life()

        expected = [(4, 24), (6, 24), (4, 23), (5, 23), (6, 23)]
        actual = life._get_cell_neighbours((5, 24))
        self.assertEqual(sorted(expected), sorted(actual))
        self.assertEqual(5, len(actual))

    def test_get_neighbours_on_right(self):
        """
        Test that we can get the correct neighbour lists on non edge cells
        """
        life = Life()

        expected = [(24, 4), (24, 6), (23, 4), (23, 5), (23, 6)]
        actual = life._get_cell_neighbours((24, 5))
        self.assertEqual(sorted(expected), sorted(actual))
        self.assertEqual(5, len(actual))

    def test_random_lifescape(self):
        """
        Try to test randomly assigning 1's to the lifescape. We could do this reliably with a seed.
        But I'm lazy, I'll just expect this test to fail sometimes (rarely).
        """
        life = Life()

        # if we do a lifescape with 30% random 1's on a 25x25 grid, we should expect 187 1's
        # I'll just check for over 50, that should be almost gauranteed
        life.randomize_lifescape(.3)
        self.assertTrue(np.sum(life.lifescape) > 50)
        self.assertFalse(np.sum(life.lifescape) > 400)

        life.randomize_lifescape(.9)
        self.assertTrue(np.sum(life.lifescape) > 400)
        self.assertFalse(np.sum(life.lifescape) < 200)

    def test_sum_neighbours(self):
        life = Life()
        arr = np.array([[0, 1, 1, 0, 0, 1],
                        [0, 1, 0, 0, 1, 1],
                        [1, 0, 0, 1, 1, 0],
                        [0, 1, 1, 0, 0, 1],
                        [0, 1, 0, 0, 1, 1],
                        [1, 0, 0, 1, 1, 0]])
        life.set_from_array(arr)

        self.assertEqual(3, life._get_sum_neighbours(life._get_cell_neighbours((1, 1))))
        self.assertEqual(2, life._get_sum_neighbours(life._get_cell_neighbours((0, 0))))
        self.assertEqual(3, life._get_sum_neighbours(life._get_cell_neighbours((5, 5))))
        self.assertEqual(4, life._get_sum_neighbours(life._get_cell_neighbours((3, 3))))


    def test_step(self):

        life = Life()
        arr_begin = np.array([[0, 1, 1, 0, 0, 1],
                              [0, 1, 0, 0, 1, 1],
                              [1, 0, 0, 1, 1, 0],
                              [0, 1, 1, 0, 0, 1],
                              [0, 1, 0, 0, 1, 1],
                              [1, 0, 0, 1, 1, 0]])


        life.set_from_array(arr_begin)
        life.do_step()

        arr_end = np.array([[0, 1, 1, 0, 1, 1],
                            [1, 1, 0, 0, 0, 1],
                            [1, 0, 0, 1, 0, 0],
                            [1, 1, 1, 0, 0, 1],
                            [1, 1, 0, 0, 0, 1],
                            [0, 0, 0, 1, 1, 1]])

        np.testing.assert_array_equal(arr_end, life.lifescape)

if __name__ == '__main__':
    unittest.main()
