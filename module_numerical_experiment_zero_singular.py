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
from multimethod import multimethod


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
                                                alpha: float, 
                                                theta: float, 
                                                num_mu_range: Iterable, 
                                                partial_dPsi0_initial: float) -> Iterator[float, float]:
    
    equation_mu_value = None

    for count, num_mu in enumerate(num_mu_range):
        count += 1
        try:
            z, dPsi, partial_dPsi0 = numerical_problem.solve_dPsi_eq(num_mu, theta, alpha, partial_dPsi0_initial)
        except RuntimeError:
            if equation_mu_value is not None:
                msg = f"RuntimeError occured during equation calculations. Parameters: alpha = {alpha:.3e}, theta = {theta:.3e}, num_mu = {num_mu:.5e}. Skiping..."
                print(msg) 
                yield equation_mu_value, partial_dPsi0
            else:
                msg = f"RuntimeError occured during equation calculations. Parameters: alpha = {alpha:.3e}, theta = {theta:.3e}, num_mu = {num_mu:.5e}. No previous results, exiting..."
                raise RuntimeError(msg)
        else:
            dp_int = numerical_problem.num_dp_int_dim(num_mu, theta, alpha, z, dPsi)
            equation_mu_value = numerical_problem.num_eq_mu(num_mu, theta, alpha, partial_dPsi0, dp_int)
            partial_dPsi0_initial = partial_dPsi0
            yield equation_mu_value, partial_dPsi0


def calculate_equation_mu_zero_singular(numerical_problem: NumericalProblem,
                                        alpha: float,
                                        theta: float,
                                        num_mu_range: Iterable[float],
                                        partial_dPsi0_initial: float) -> MetaData:
    
    equation_mu_partial_dPsi0_values = calculate_equation_mu_partial_dPsi0_values(numerical_problem, alpha, theta, num_mu_range, partial_dPsi0_initial)
    gen = zip(pairwise(num_mu_range), pairwise(equation_mu_partial_dPsi0_values))
    num_mu_zero_singular = []
    partial_dPsi0_zero_singular = []

    for (_, num_mu_current), ((equation_mu_value_previous, _), (equation_mu_value_current, partial_dPsi0_current)) in gen:
        if equation_mu_value_previous * equation_mu_value_current < 0:
                num_mu_zero_singular.append(num_mu_current)
                partial_dPsi0_zero_singular.append(partial_dPsi0_current)
        if len(num_mu_zero_singular) == 2:
            break

    if len(num_mu_zero_singular) == 1:
        print("Warning! Only zero point was found. Assuming num_mu_singular = num_mu_zero.")
        num_mu_zero_singular.extend(num_mu_zero_singular)
    elif len(num_mu_zero_singular) != 2:
        raise RuntimeError(f"No points were found. Parameters: alpha = {alpha:.3e}, theta = {theta:.3e}, num_mu = {num_mu_current:.5e}. Exiting...")
    metadata = MetaData(alpha, theta, *num_mu_zero_singular, *partial_dPsi0_zero_singular)
    return metadata    


def __verify_inputs(var: Any) -> RealNumpyArray:
    if not all(isinstance(element, float) for element in var):
            raise TypeError(f"theta_range and alpha_range must be either Iterable[float] or float")
    return np.asarray(var)


def __estimate_interval(metadata_current: MetaData, metadata_previous: MetaData, param: str) -> tuple[float, float]:
    names = ("num_mu_zero", "num_mu_singular", param)
    num_mu_zero_current, num_mu_singular_current, param_current = (getattr(metadata_current, name) for name in names)
    num_mu_zero_previous, num_mu_singular_previous, param_previous = (getattr(metadata_previous, name) for name in names)

    dparam = param_current - param_previous
    dmu_dalpha_zero = (num_mu_zero_current - num_mu_zero_previous) / dparam
    dmu_dalpha_singular = (num_mu_singular_current - num_mu_singular_previous) / dparam

    zero_cond = -min(dmu_dalpha_zero, dmu_dalpha_singular)
    singular_cond = max(dmu_dalpha_zero, dmu_dalpha_singular)

    num_mu_left = 2 * num_mu_zero_current - num_mu_singular_current - np.heaviside(zero_cond, 0.5) * dparam
    num_mu_right = 2 * num_mu_singular_current - num_mu_zero_current + np.heaviside(singular_cond, 0.5) * dparam
    return num_mu_left, num_mu_right


@multimethod
def __getting_statistics(numerical_problem: NumericalProblem,
                    alpha_range: Iterable | float,
                    theta_range: Iterable | float,
                    num_mu_range: Iterable,
                    N_num_mu_points: int,
                    verbose: bool = False,
                    show_progress: bool = False) -> Iterator:
    
    parameters_passed = locals()
    info = ' '.join(f"{type(value).__name__}" for _, value in parameters_passed.items())
    msg = f"Could not find signature for getting_statistics: <{info}> "
    raise NotImplementedError(msg)

@multimethod
def __getting_statistics(numerical_problem: NumericalProblem,
                    alpha_range: Iterable,
                    theta: float,
                    num_mu_range: Iterable,
                    N_num_mu_points: int,
                    verbose: bool = False,
                    show_progress: bool = False) -> Iterator:
    _DELTA = 0.15

    alpha_range = __verify_inputs(alpha_range)
    num_mu_range = np.linspace(num_mu_range[0], num_mu_range[-1], N_num_mu_points)

    if show_progress:
        p_bar_len = len(alpha_range)
        p_bar = tqdm(total = p_bar_len)
        logger = tqdm.write
        update = p_bar.update
    else:
        p_bar = nullcontext()
        update = lambda *args, **kwargs: ...
        logger = print

    metadata_current = None
    metadata_previous = None

    partial_dPsi0_initial = numerical_problem.estimate_partial_dPsi0(num_mu_range[0], theta, alpha_range[0])
    with p_bar:
        for alpha in alpha_range:
            metadata = calculate_equation_mu_zero_singular(numerical_problem, alpha, theta, num_mu_range, partial_dPsi0_initial)
            num_mu_left, num_mu_right = metadata.num_mu_zero, metadata.num_mu_singular

            if metadata_current is None:
                metadata_current = metadata
            else:
                metadata_previous = metadata_current
                metadata_current = metadata

            try:
                num_mu_left, num_mu_right = __estimate_interval(metadata_current, metadata_previous, "alpha")
            except AttributeError:
                num_mu_zero, num_mu_singular = metadata.num_mu_zero, metadata.num_mu_singular
                num_mu_left, num_mu_right = num_mu_zero - _DELTA, num_mu_singular + _DELTA

            num_mu_range = np.linspace(num_mu_left, num_mu_right, N_num_mu_points)
            if verbose:
                info = f"alpha = {alpha:.5e}, new interval: ({num_mu_left}, {num_mu_right})"
                logger(info)
            partial_dPsi0_initial = metadata.partial_dPsi0_zero
            _ = update()
            yield metadata

@multimethod
def __getting_statistics(numerical_problem: NumericalProblem,
                    alpha: float,
                    theta_range: Iterable,
                    num_mu_range: Iterable,
                    N_num_mu_points: int,
                    verbose: bool = False,
                    show_progress: bool = False) -> Iterator:
    _DELTA = 0.15

    theta_range = __verify_inputs(theta_range)
    num_mu_range = np.linspace(num_mu_range[0], num_mu_range[-1], N_num_mu_points)

    if show_progress:
        p_bar_len = len(theta_range)
        p_bar = tqdm(total = p_bar_len)
        logger = p_bar.write
        update = p_bar.update
    else:
        p_bar = nullcontext()
        update = lambda *args, **kwargs: ...
        logger = print

    metadata_current = None
    metadata_previous = None

    partial_dPsi0_initial = numerical_problem.estimate_partial_dPsi0(num_mu_range[0], theta_range[0], alpha)
    with p_bar:
        for theta in theta_range:
            metadata = calculate_equation_mu_zero_singular(numerical_problem, alpha, theta, num_mu_range, partial_dPsi0_initial)
            num_mu_left, num_mu_right = metadata.num_mu_zero, metadata.num_mu_singular

            if metadata_current is None:
                metadata_current = metadata
            else:
                metadata_previous = metadata_current
                metadata_current = metadata

            try:
                num_mu_left, num_mu_right = __estimate_interval(metadata_current, metadata_previous, "theta")
            except AttributeError:
                num_mu_zero, num_mu_singular = metadata.num_mu_zero, metadata.num_mu_singular
                num_mu_left, num_mu_right = num_mu_zero - _DELTA, num_mu_singular + _DELTA

            num_mu_range = np.linspace(num_mu_left, num_mu_right, N_num_mu_points)
            if verbose:
                info = f"alpha = {alpha:.5e}, new interval: ({num_mu_left}, {num_mu_right})"
                logger(info)
            partial_dPsi0_initial = metadata.partial_dPsi0_zero
            _ = update()
            yield metadata


def getting_statistics(numerical_problem: NumericalProblem,
                    alpha_range: Iterable[float] | float,
                    theta_range: Iterable[float] | float,
                    num_mu_range: Iterable[float],
                    N_num_mu_points: int,
                    verbose: bool = True,
                    show_progress: bool = False) -> Iterator[MetaData]:
    gen = __getting_statistics(numerical_problem, alpha_range, theta_range, num_mu_range, N_num_mu_points, verbose, show_progress)
    for metadata in gen:
        yield metadata
