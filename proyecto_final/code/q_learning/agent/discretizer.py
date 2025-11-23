import numpy as np

def aggregate_and_discretize_strip(strip, num_levels):
    mean_pixel_value = np.mean(strip)
    bin_size = 256 / num_levels
    discretized_level = int(mean_pixel_value / bin_size)
    return min(discretized_level, num_levels - 1)


def convert_state_vector_to_int(state_vector, num_levels):
    index = 0
    for i, level in enumerate(state_vector):
        index += level * (num_levels ** (len(state_vector) - 1 - i))
    return int(index)
