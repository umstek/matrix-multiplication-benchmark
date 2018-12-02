import random
from typing import List

Matrix = List[List[float]]


def init_matrix(size: int) -> Matrix:
    return list(map(lambda _: [0.0] * size, range(size)))


def generate_matrix(size: int) -> Matrix:
    return list(map(
        lambda _: list(map(
            lambda _: random.uniform(-1000, 1000),  # Range of the number does not matter as this is floating point.
            range(size)
        )),
        range(size)
    ))
