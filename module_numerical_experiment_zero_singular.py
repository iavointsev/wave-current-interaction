from module_utils import *
from module_preparations import RealNumpyArray, NumericalParameters, NumericalProblem, NumericalSolution
import numpy as np
from tqdm import tqdm
import sympy as sym
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import NamedTuple, Any
from collections.abc import Callable, Iterable, Iterator
from itertools import pairwise, product
from contextlib import nullcontext


class MetaData(NamedTuple):
    alpha: float
    theta: float
    num_mu_zero: float
    num_mu_singular: float
    partial_dPsi0_zero: float
    partial_dPsi0_singular: float
    def __str__(self):
        return "{:.14f}\t{:.14f}\t{:.14f}\t{:.14f}\t{:.14f}\t{:.14f}".format(*self)


def calculate_equation_mu_partial_dPsi0_values(numerical_problem: NumericalProblem, 
                                                theta: float, 
                                                alpha: float, 
                                                num_mu_range: Iterable, 
                                                partial_dPsi0_initial: float) -> Iterator[float, float]:
    
    equation_mu_value = None

    for count, num_mu in enumerate(num_mu_range):
        count += 1
        try:
            z, dPsi, partial_dPsi0 = numerical_problem.solve_dPsi_eq(num_mu, theta, alpha, partial_dPsi0_initial)
        except RuntimeError:
            if equation_mu_value is not None:
                msg = f"RuntimeError occured when alpha = {alpha:.3e}, theta = {theta:.3e}, num_mu = {num_mu:.5e}. Skiping..."
                print(msg) 
                yield equation_mu_value, partial_dPsi0
            else:
                msg = f"RuntimeError occured when alpha = {alpha:.3e}, theta = {theta:.3e}, num_mu = {num_mu:.5e}. No previous results, exiting..."
                raise RuntimeError(msg)
        else:
            dp_int = numerical_problem.num_dp_int_dim(num_mu, theta, alpha, z, dPsi)
            equation_mu_value = numerical_problem.num_eq_mu(num_mu, theta, alpha, partial_dPsi0, dp_int)
            partial_dPsi0_initial = partial_dPsi0
            yield equation_mu_value, partial_dPsi0


def calculate_equation_mu_zero_singular(numerical_problem: NumericalProblem,
                                        theta: float,
                                        alpha: float,
                                        num_mu_range: Iterable[float],
                                        partial_dPsi0_initial: float) -> MetaData:
    
    equation_mu_partial_dPsi0_values = calculate_equation_mu_partial_dPsi0_values(numerical_problem, theta, alpha, num_mu_range, partial_dPsi0_initial)
    gen = zip(pairwise(num_mu_range), pairwise(equation_mu_partial_dPsi0_values))
    num_mu_zero_singular = []
    partial_dPsi0_zero_singular = []

    for (_, num_mu_current), ((equation_mu_value_previous, _), (equation_mu_value_current, partial_dPsi0_current)) in gen:
        if equation_mu_value_previous * equation_mu_value_current < 0:
                num_mu_zero_singular.append(num_mu_current)
                partial_dPsi0_zero_singular.append(partial_dPsi0_current)
        if len(num_mu_zero_singular) == 2:
            break

    if len(num_mu_zero_singular) != 2:
        raise RuntimeError(f"Points not found. Exiting,,,")
    metadata = MetaData(alpha, theta, *num_mu_zero_singular, *partial_dPsi0_zero_singular)
    return metadata    


def _verify_inputs(var: Any) -> Iterable[float]:
    if isinstance(var, Iterable): 
        if not all(isinstance(theta, float) for theta in var):
            raise TypeError(f"theta_range and alpha_range must be either Iterable[float] or float")
    else:
        try:
            var = [var]
        except:
            raise TypeError(f"Could not convert {var.__class__.__name__} into list[float]")
    return var


def getting_statistics(numerical_problem: NumericalProblem,
                    theta_range: Iterable[float] | float,
                    alpha_range: Iterable[float] | float,
                    num_mu_range: Iterable[float],
                    N_num_mu_points: int = 256,
                    show_progress: bool = False) -> Iterator[MetaData]:
    
    _DELTA = 0.15
    
    theta_range = _verify_inputs(theta_range)
    alpha_range = _verify_inputs(alpha_range)

    num_mu_range = np.linspace(num_mu_range[0], num_mu_range[-1], N_num_mu_points)

    if show_progress:
        p_bar_len = len(alpha_range) * len(theta_range)
        p_bar = tqdm(total = p_bar_len)
        logger = tqdm.write
        update = p_bar.update
    else:
        p_bar = nullcontext()
        update = lambda *args, **kwargs: ...
        logger = print

    with p_bar:
        for alpha in alpha_range:
            partial_dPsi0_initial = numerical_problem.estimate_partial_dPsi0(num_mu_range[0], theta_range[0], alpha)
            for theta in theta_range:
                metadata = calculate_equation_mu_zero_singular(numerical_problem, theta, alpha, num_mu_range, partial_dPsi0_initial)
                num_mu_left, num_mu_right = metadata.num_mu_zero, metadata.num_mu_singular
                num_mu_range = np.linspace(num_mu_left - _DELTA, num_mu_right + _DELTA, N_num_mu_points)
                partial_dPsi0_initial = metadata.partial_dPsi0_zero
                yield metadata
                _ = update()
                

