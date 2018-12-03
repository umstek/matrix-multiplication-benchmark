import random
import timeit
# noinspection PyUnresolvedReferences
import os

import numpy as np

# noinspection PyUnresolvedReferences
from common import init_matrix, generate_matrix
from serial import multiply_matrices_serial
from parallel import multiply_matrices_parallel
from parallel_opt import multiply_matrices_parallel_opt
from cuda import multiply_matrices_cuda


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


def test_cuda(size):
    setup = f'''
np.random.seed()

m1 = np.random.rand({size}, {size})
m2 = np.random.rand({size}, {size})
m3 = np.zeros(shape=({size}, {size}))
'''

    script = 'multiply_matrices_cuda(m1, m2, m3)'
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
    mtc3 = np.zeros(shape=(3, 3))

    multiply_matrices_parallel(mtp1, mtp2, mtp3)
    print(mtp3, "\n--")

    multiply_matrices_parallel_opt(mtp1, mtp2, mtpo3)
    print(mtpo3, "\n--")

    multiply_matrices_cuda(mtp1, mtp2, mtc3)
    print(mtc3)


if __name__ == "__main__":
    sanity_checks()

    sz = '1000'  # Size of the matrix

    # Number of samples + 1 inside range
    # 1st calculation is omitted, this is frequently causes a warm-up and is an outlier
    # time_serial = [test_serial(sz) for i in range(4)][1:]
    # print(np.mean(time_serial), np.std(time_serial))
    # time_parallel = [test_parallel(sz) for i in range(8)][1:]
    # print(np.mean(time_parallel), np.std(time_parallel))
    # time_parallel_opt = [test_parallel_opt(sz) for i in range(20)][1:]
    # print(np.mean(time_parallel_opt), np.std(time_parallel_opt))
    time_cuda = [test_cuda('2000') for i in range(10)][1:]
    print(np.mean(time_cuda), np.std(time_cuda))
