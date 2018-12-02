import random
import timeit
import numpy as np
# noinspection PyUnresolvedReferences
from common import init_matrix, generate_matrix
# noinspection PyUnresolvedReferences
from serial import multiply_matrices_serial
# noinspection PyUnresolvedReferences
from parallel import multiply_matrices_parallel

setup = '''
m1 = np.random.rand(2000, 2000)
m2 = np.random.rand(2000, 2000)
m3 = np.zeros(shape=(2000, 2000))
'''


def test_serial():
    random.seed(0)  # TODO Remove

    script = 'multiply_matrices_serial(m1, m2, m3)'
    return timeit.timeit(script, setup=setup, number=5, globals=globals())


def test_parallel():
    random.seed(0)  # TODO Remove

    setup = '''
m1 = np.random.rand(200, 200)
m2 = np.random.rand(200, 200)
m3 = np.zeros(shape=(200, 200))
    '''
    # multiply_matrices_parallel(m1, m2, m3)
    # print(m3)
    # mat = init_matrix(200)
    # multiply_matrices_parallel(generate_matrix(200), generate_matrix(200), mat)
    script = 'multiply_matrices_parallel(m1, m2, m3)'
    return timeit.timeit(script, setup=setup, number=5, globals=globals())


if __name__ == "__main__":
    time_serial = test_serial()
    print(time_serial)
    time_parallel = test_parallel()
    print(time_parallel)
