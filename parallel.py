from operator import mul
from numba import njit

from common import Matrix


@njit(parallel=True)
def multiply_matrices_parallel(mat1: Matrix, mat2: Matrix, result: Matrix) -> None:
    size = len(result)

    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += mat1[i][k] * mat2[k][j]
