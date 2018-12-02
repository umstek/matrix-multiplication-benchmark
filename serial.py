from common import Matrix


def multiply_matrices_serial(mat1: Matrix, mat2: Matrix, result: Matrix) -> None:
    size = len(result)

    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += mat1[i][k] * mat2[k][j]

    # Initializing result matrix is not a part of the computation
