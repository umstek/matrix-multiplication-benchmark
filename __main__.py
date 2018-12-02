import random
import timeit

import numpy as np

# noinspection PyUnresolvedReferences
from common import init_matrix, generate_matrix
from serial import multiply_matrices_serial
from parallel import multiply_matrices_parallel
from parallel_opt import multiply_matrices_parallel_opt


def test_serial():
    setup = '''
random.seed()

m1 = generate_matrix(200)
m2 = generate_matrix(200)
m3 = init_matrix(200)
'''

    script = 'multiply_matrices_serial(m1, m2, m3)'
    return timeit.timeit(script, setup=setup, number=1, globals=globals())


def test_parallel():
    setup = '''
import os


os.environ['NUMBA_OPT'] = '0'
numpy.random.seed()

m1 = np.random.rand(2000, 2000)
m2 = np.random.rand(2000, 2000)
m3 = np.zeros(shape=(2000, 2000))
'''

    script = 'multiply_matrices_parallel(m1, m2, m3)'
    return timeit.timeit(script, setup=setup, number=1, globals=globals())


def test_parallel_opt():
    setup = '''
import os


os.environ['NUMBA_OPT'] = '3'
numpy.random.seed()
   
m1 = np.random.rand(2000, 2000)
m2 = np.random.rand(2000, 2000)
m3 = np.zeros(shape=(2000, 2000))
'''

    script = 'multiply_matrices_parallel_opt(m1, m2, m3)'
    return timeit.timeit(script, setup=setup, number=1, globals=globals())


def sanity_checks():
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

    time_serial = test_serial()
    print(time_serial)
    time_parallel = test_parallel()
    print(time_parallel)
    time_parallel_opt = test_parallel_opt()
    print(time_parallel_opt)
