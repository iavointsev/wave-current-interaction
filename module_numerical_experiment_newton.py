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
    num_mu: float
    partial_dPsi0: float
    def __str__(self):
        return "{:.14f}\t{:.14f}\t{:.14f}\t{:.14f}".format(*self)
    

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
    


def calculate_metadata(numerical_problem: NumericalProblem,  
                        theta: float,
                        alpha: float,
                        num_mu_initial: float,
                        partial_dPsi0_initial: float) -> MetaData:

    try:
        solution = numerical_problem.solve_mu_eq_newton(num_mu_initial, theta, alpha, partial_dPsi0_initial)
    except RuntimeError:
        msg = (f'Runtime error occured with theta = {theta:.5e}, alpha = {alpha:.5e}, ' + 
            f'num_mu_initial = {num_mu_initial:.5f}, partial_dPsi0_initial = {partial_dPsi0_initial:.5f}')
        raise RuntimeError(msg)
    else:
        num_mu, partial_dPsi0 = solution.num_mu, solution.partial_dPsi0
        return MetaData(alpha, theta, num_mu, partial_dPsi0)
    

def getting_statistics(numerical_problem: NumericalProblem,
                       theta_range: Iterable[float] | float,
                       alpha_range: Iterable[float] | float,
                       num_mu_initial: float,
                       show_progress: bool = False,
                       concurrent_workers: int = 1) -> Iterator[MetaData]:
    
    theta_range = _verify_inputs(theta_range)
    alpha_range = _verify_inputs(alpha_range)
    
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
            partial_dPsi0_initial = numerical_problem.estimate_partial_dPsi0(num_mu_initial, theta_range[0], alpha)
            for theta in theta_range:
                metadata = calculate_metadata(numerical_problem, theta, alpha, num_mu_initial, partial_dPsi0_initial)
                num_mu_initial = metadata.num_mu
                partial_dPsi0_initial = metadata.partial_dPsi0
                yield metadata
                _ = update()


            
