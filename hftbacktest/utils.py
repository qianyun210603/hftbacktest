from numba import njit

@njit
def get_time_tuple(timestamp, sub_second_factor=1000):
    timestamp %= 86400*sub_second_factor
    timestamp, millisecond = divmod(timestamp, sub_second_factor)
    millisecond /= sub_second_factor / 1000
    timestamp, second = divmod(timestamp, 60)
    hour, minute = divmod(timestamp, 60)
    return hour, minute, second, millisecond