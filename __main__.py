import random
import timeit
# noinspection PyUnresolvedReferences
import os
import math

import numpy as np

# noinspection PyUnresolvedReferences
from common import init_matrix, generate_matrix
from serial import multiply_matrices_serial
from parallel import multiply_matrices_parallel
from parallel_opt import multiply_matrices_parallel_opt


def test_serial(size):
    setup = f'''
random.seed()

m1 = generate_matrix({size})
m2 = generate_matrix({size})
m3 = init_matrix({size})
'''

    script = 'multiply_matrices_serial(m1, m2, m3)'
    return timeit.timeit(script, setup=setup, number=1, globals=globals())


def test_parallel(size):
    setup = f'''
import os


os.environ['NUMBA_OPT'] = '0'
np.random.seed()

m1 = np.random.rand({size}, {size})
m2 = np.random.rand({size}, {size})
m3 = np.zeros(shape=({size}, {size}))
'''

    script = 'multiply_matrices_parallel(m1, m2, m3)'
    return timeit.timeit(script, setup=setup, number=1, globals=globals())


def test_parallel_opt(size):
    setup = f'''
import os


os.environ['NUMBA_OPT'] = '3'
np.random.seed()
   
m1 = np.random.rand({size}, {size})
m2 = np.random.rand({size}, {size})
m3 = np.zeros(shape=({size}, {size}))
'''

    script = 'multiply_matrices_parallel_opt(m1, m2, m3)'
    return timeit.timeit(script, setup=setup, number=1, globals=globals())


def sanity_checks():
    # To visually test whether implementations are correct
    random.seed()

    mt1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mt2 = [[9, 8, 7, ], [6, 5, 4], [3, 2, 1]]
    mt3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    multiply_matrices_serial(mt1, mt2, mt3)
    print(mt3, "\n--")

    mtp1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mtp2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    mtp3 = np.zeros(shape=(3, 3))
    mtpo3 = np.zeros(shape=(3, 3))

    multiply_matrices_parallel(mtp1, mtp2, mtp3)
    print(mtp3, "\n--")

    multiply_matrices_parallel_opt(mtp1, mtp2, mtpo3)
    print(mtpo3, "\n--")


if __name__ == "__main__":
    sanity_checks()

    # time_serial = [test_serial('200') for i in range(10)]
    # print(time_serial, np.mean(time_serial), np.std(time_serial))
    time_parallel = [test_parallel('1200') for i in range(5)][1:]
    print(np.mean(time_parallel), np.std(time_parallel))
    # time_parallel_opt = [test_parallel_opt('1000') for i in range(25)][1:]
    # print(np.mean(time_parallel_opt), np.std(time_parallel_opt))
