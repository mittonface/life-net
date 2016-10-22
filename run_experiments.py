import numpy as np
from gol_generator import Life
from life_learner import build_life_nn
import matplotlib.pyplot as plt


def simple_solution(num_steps=50):
    life = Life()
    life.randomize_lifescape()


    for i in range(num_steps):
        life.do_step()

        ax = plt.axes()
        ax.matshow(life.lifescape)
        ax.set_axis_off()
        plt.savefig('simple/life%s.png' % str(i).zfill(3))


def default_nn_solution(num_steps=50):
    life = Life()
    life.randomize_lifescape()

    nn, training_error = build_life_nn()

default_nn_solution()
