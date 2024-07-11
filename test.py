import numpy as np
from typing import NamedTuple, Any
from collections.abc import Callable, Iterable, Iterator


class MetaData(NamedTuple):
    alpha: float
    theta: float
    num_mu_zero: float
    num_mu_singular: float
    partial_dPsi0_zero: float
    partial_dPsi0_singular: float
    def __str__(self):
        return "{:.14f}\t{:.14f}\t{:.14f}\t{:.14f}\t{:.14f}\t{:.14f}".format(*self)


def __getting_statistics(numerical_problem: MetaData,
                    alpha_range: Iterable | float,
                    theta_range: Iterable | float,
                    num_mu_range: Iterable,
                    N_num_mu_points: int,
                    show_progress: bool = False) -> Iterator:
    
    parameters_passed = locals()
    print(parameters_passed)
    info = ' '.join(f"{type(value).__name__}" for _, value in parameters_passed.items())
    msg = f"Could not find signature for getting_statistics: <{info}> "
    raise NotImplementedError(msg)


if __name__ == "__main__":
    metadata = MetaData(1, 2, 3, 4, 5, 6)
    alpha_range = np.linspace(0, 1, 10)
    theta_range = [1, 2, 3]
    num_mu_range = (1, 2)
    N_num_mu_points = 10
    show_progress = True

    __getting_statistics(metadata, alpha_range, theta_range, num_mu_range, N_num_mu_points)


