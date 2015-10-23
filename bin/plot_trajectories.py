#!/usr/bin/env python

from __future__ import division

from collections import import namedtuple
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import import LogNorm
from matplotlib.ticker import import MultipleLocator, FormatStrFormatter
import multiprocessing
import numpy as np

from ep.helpers import import map_trajectory
import ep.plot
from ep.waveguide import import DirichletPositionDependentLoss, Dirichlet


ep.plot.get_defaults()
colors, parula, _ = ep.plot.get_colors()


def plot_trajectories_energy():
    pass


if __name__ == '__main__':
     plot_trajectories_energy()
