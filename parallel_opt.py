from numba import njit


@njit(parallel=True)
def multiply_matrices_parallel_opt(mat1, mat2, result):
    size = len(result)

    for i in range(size):
        for k in range(size):
            for j in range(size):
                result[i][j] += mat1[i][k] * mat2[k][j]
