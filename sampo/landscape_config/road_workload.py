def intensity(vehicle_num: int, length: float, max_velocity: float):
    return vehicle_num * max_velocity / length


def static_workload(vehicle_num: int, length: float, max_velocity: float, bandwidth: float):
    """
    Calculate rate of theoretical road workload
    :param bandwidth:
    :param vehicle_num:
    :param length:
    :param max_velocity:
    :return:
    """
    return intensity(vehicle_num, length, max_velocity) / bandwidth
