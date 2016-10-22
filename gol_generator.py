import numpy as np
import random


class Life:

    def __init__(self, size=(25, 25)):
        self.lifescape = np.zeros(size, dtype=np.int)

    def set_from_array(self, A):
        self.lifescape = A

    def do_step(self):
        # I could do this much faster with numpy

        # get a list of the cells to make dead and the ones to make alive
        make_alive = []
        make_dead = []
        for i in range(self.lifescape.shape[0]):
            for j in range(self.lifescape.shape[1]):
                if self._get_sum_neighbours(self._get_cell_neighbours((i, j))) < 2:
                    make_dead.append((i, j))
                elif self._get_sum_neighbours(self._get_cell_neighbours((i, j))) > 3:
                    make_dead.append((i, j))
                elif self._get_sum_neighbours(self._get_cell_neighbours((i, j))) == 3:
                    make_alive.append((i, j))

        # make them alive or dead
        for i in range(self.lifescape.shape[0]):
            for j in range(self.lifescape.shape[1]):
                if (i, j) in make_dead:
                    self.lifescape[i][j] = 0
                if (i, j) in make_alive:
                    self.lifescape[i][j] = 1


    def randomize_lifescape(self, percent_activated=0.3):
        rfunc = np.vectorize(_set_living)
        self.lifescape = rfunc(self.lifescape, percent_activated)


    def _get_cell_neighbours(self, location):
        """
        Returns a list of locations corresponding to the given locations valid neighbours
        """
        neighbours = [
            (location[0]-1, location[1]-1),     # TL
            (location[0], location[1]-1),       # T
            (location[0]+1, location[1]-1),     # TR
            (location[0]-1, location[1]),       # L
            (location[0]+1, location[1]),       # R
            (location[0]-1, location[1]+1),     # BL
            (location[0], location[1]+1),       # B
            (location[0]+1, location[1]+1),     # BR
        ]

        return self._filter_bad_cells(neighbours)

    def _get_sum_neighbours(self, neighbours):
        sum = 0
        for n in neighbours:
            sum += self.lifescape[n[0]][n[1]]
        return sum

    def _filter_bad_cells(self, neighbours):
        """
        Filters all of the invalid locations out of a list of locations
        """
        good_neighbours = []

        for n in neighbours:
            if not self._is_bad_location(n):
                good_neighbours.append(n)

        return good_neighbours

    def _is_bad_location(self, location):
        """
        Detects whether a a given location is valid
        """
        return location[0] < 0 \
               or location[1] < 0 \
               or location[0] > self.lifescape.shape[0] - 1 \
               or location[1] > self.lifescape.shape[1] - 1


def _set_living(x, p):
    r = random.random()
    if r < p:
        return 1
    else:
        return 0
