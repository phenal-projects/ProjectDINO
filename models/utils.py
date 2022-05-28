def linear_annealing_with_plateau(start: float, end: float, T: int):
    delta = end - start / T
    i = 0
    while True:
        i += 1
        if i > T:
            yield end
        else:
            yield start + delta * i
