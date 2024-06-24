def max_fill(mat_max: int, mat_available: int):
    return mat_max - mat_available


def necessary_fill(mat_count: int, mat_available: int, mat_max: int):
    return mat_count + mat_available - mat_max
