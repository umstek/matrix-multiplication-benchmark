import math
from numba import cuda, guvectorize


@cuda.jit()
def matmul(mat1, mat2, result):
    i, j = cuda.grid(2)
    if i < result.shape[0] and j < result.shape[1]:
        tmp = 0.
        for k in range(result.shape[1]):
            tmp += mat1[i][k] * mat2[k][j]
        result[i][j] = tmp


def multiply_matrices_cuda(mat1, mat2, result):
    size = len(result)

    mat1_d = cuda.to_device(mat1)
    mat2_d = cuda.to_device(mat2)

    result_d = cuda.device_array((size, size))
    threadsperblock = (16, 16)
    blockspergrid_x = int(math.ceil(mat1.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(mat2.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    matmul[blockspergrid, threadsperblock](mat1_d, mat2_d, result_d)

    result_d.copy_to_host(result)
