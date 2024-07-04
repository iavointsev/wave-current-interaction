from module_utils import *
from module_symbols import *
import sympy as sym
import numpy as np
from scipy.optimize import newton
from scipy.integrate import solve_ivp, simpson
from typing import NamedTuple
from collections.abc import Callable
import numpy.typing as npt
import dill
from pathlib import Path


RealNumpyArray = npt.NDArray[np.float64]


def get_stokes_drift(verbose: bool = False):

    # Us0
    verbose_print(verbose, 'Initial wave: ')
    verbose_display(verbose, sym.Eq(sym_phi0_symbol, sym_phi0))
    sym_u0x, sym_u0y, sym_u0z = (sym_phi0.diff(var) for var in (sym_x, sym_y, sym_z))

    sym_Us0x = (sym.I * sym_u0y / sym_omega * sym_u0x.conjugate()).diff(sym_y) + \
                (sym.I * sym_u0z / sym_omega * sym_u0x.conjugate()).diff(sym_z)
    sym_Us0x = sym_Us0x / 2
    sym_Us0y = 0
    sym_Us0x = sym_Us0x.subs(sym_psi0, - sym.I * sym_omega * sym_h0) #!!!!!!!!!
    sym_Us0x = sym_Us0x.simplify()
    
    verbose_display(verbose, sym.Eq(sym.Symbol(r'U^{s0}_x'), sym_Us0x))

    # dUs
    verbose_print(verbose, 'Oblique wave: ')
    verbose_display(verbose, sym.Eq(sym_dphi_symbol, sym_dphi))
    sym_dux, sym_duy, sym_duz = (sym_dphi.diff(var) for var in (sym_x, sym_y, sym_z))

    sym_dUsx = (sym.I * sym_duy / sym_omega * (sym_u0x).conjugate()).diff(sym_y) + \
                (sym.I * sym_duz / sym_omega * (sym_u0x).conjugate()).diff(sym_z) + \
                ((sym.I * sym_u0y / sym_omega).conjugate() * sym_dux).diff(sym_y) + \
                ((sym.I * sym_u0z / sym_omega).conjugate() * sym_dux).diff(sym_z)
    sym_dUsx = sym_dUsx / 2
    sym_dUsy = (sym.I * sym_dux / sym_omega * (sym_u0y).conjugate()).diff(sym_x) + \
                (sym.I * sym_duz / sym_omega * (sym_u0y).conjugate()).diff(sym_z) + \
                ((sym.I * sym_u0x / sym_omega).conjugate() * sym_duy).diff(sym_x) + \
                ((sym.I * sym_u0z / sym_omega).conjugate() * sym_duy).diff(sym_z)
    sym_dUsy = sym_dUsy / 2
    
    sym_dUsx = sym_dUsx.subs(sym_psi0, - sym.I * sym_omega * sym_h0)
    sym_dUsy = sym_dUsy.subs(sym_psi0, - sym.I * sym_omega * sym_h0)

    sym_dUsx = sym_dUsx.simplify()
    sym_dUsy = sym_dUsy.simplify()

    verbose_display(verbose, sym.Eq(sym.Symbol(r'\delta U^s_x'), sym_dUsx))
    verbose_display(verbose, sym.Eq(sym.Symbol(r'\delta U^s_y'), sym_dUsy))

    return (sym_Us0x, sym_Us0y, 0), (sym_alpha * sym_dUsx, sym_alpha * sym_dUsy, 0)


def get_vortex_force(verbose: bool = False):

    sym_Us0_vec, sym_dUs_vec = get_stokes_drift(verbose = verbose)
     
    sym_vec_O0 = (0, sym_O0, 0)

    sym_dfV_1 = vector_mult(sym_Us0_vec, sym_vec_dO) 
    sym_dfV_2 = vector_mult(sym_dUs_vec, sym_vec_O0)

    sym_dfV = tuple(map(lambda x, y: x + y, sym_dfV_1, sym_dfV_2))
    sym_dfV = simplify_vector(sym_dfV)
    sym_curl_dfV = curl(sym_dfV)
    sym_curl_dfV = simplify_vector(sym_curl_dfV)

    verbose_display(verbose, sym.Eq(sym.Symbol(r'\delta{f^V}_x'), sym_dfV[0]))
    verbose_display(verbose, sym.Eq(sym.Symbol(r'\delta{f^V}_y'), sym_dfV[1]))
    verbose_display(verbose, sym.Eq(sym.Symbol(r'\delta{f^V}_z'), sym_dfV[2]))
    verbose_display(verbose, sym.Eq(sym.Symbol(r'\big(\operatorname{curl}\delta{f^V}\big)_x'), sym_curl_dfV[0]))
    verbose_display(verbose, sym.Eq(sym.Symbol(r'\big(\operatorname{curl}\delta{f^V}\big)_y'), sym_curl_dfV[1]))
    verbose_display(verbose, sym.Eq(sym.Symbol(r'\big(\operatorname{curl}\delta{f^V}\big)_z'), sym_curl_dfV[2]))

    return sym_dfV, sym_curl_dfV


def get_equation_dPsi(verbose: bool = False):

    sym_dfV, sym_curl_dfV = get_vortex_force(verbose = verbose)
    sym_dfVx = sym_dfV[0]
    sym_curl_dfVx = sym_curl_dfV[0]

    laplas = lambda expr: expr.diff(sym_x, 2) + expr.diff(sym_y, 2) + expr.diff(sym_z, 2)

    sym_nu = 0

    # equation (38)
    sym_equation_dOx = (sym_lambda * sym_dOx - sym_nu * laplas(sym_dOx) + sym_V0_symbol * sym_dOx.diff(sym_x)) - \
                        sym_O0 * (sym_dOz + sym_dVx.diff(sym_y)) - sym_curl_dfVx
    sym_equation_dOx = sym_equation_dOx.simplify()

    # equation (39)
    sym_equation_dVx = (sym_lambda * sym_dVx - sym_nu * laplas(sym_dVx) + sym_V0_symbol * sym_dVx.diff(sym_x)) + \
                        sym_dVz * sym_O0 - sym_dfVx
    sym_equation_dVx = sym_equation_dVx.simplify()

    verbose_print(verbose, "Initial equations: ")
    verbose_display(verbose, sym.Eq(sym_equation_dVx, 0))
    verbose_display(verbose, sym.Eq(sym_equation_dOx, 0))

    sym_dVx_dVz_ratio = sym.solve(sym_equation_dVx, sym_dVx_symbol)[0]
    sym_dVy_to_dPsi = sym_dPsi.diff(sym_z)
    sym_dVz_to_dPsi = -sym.I * sym_theta * sym_dPsi
    sym_dVx_to_dPsi = sym_dVx_dVz_ratio.subs(sym_dVz_symbol, sym_dVz_to_dPsi)

    verbose_display(verbose, sym.Eq(sym_dVx_symbol, sym_dVx_to_dPsi))
    verbose_display(verbose, sym.Eq(sym_dVy_symbol, sym_dVy_to_dPsi))
    verbose_display(verbose, sym.Eq(sym_dVz_symbol, sym_dVz_to_dPsi))

    substitutions_dV_to_dPsi = ((sym_dVx_symbol, sym_dVx_to_dPsi), (sym_dVy_symbol, sym_dVy_to_dPsi), (sym_dVz_symbol, sym_dVz_to_dPsi))
    sym_equation_dPsi = substitute(sym_equation_dOx, substitutions_dV_to_dPsi)
    sym_equation_dPsi = sym_equation_dPsi.simplify()

    over_prefactor = sym.exp(-sym.I * sym_theta * sym_y - sym_lambda * sym_t)
    sym_equation_dPsi = sym_equation_dPsi * over_prefactor
    sym_equation_dPsi = sym_equation_dPsi.simplify()

    sym_equation_dPsi = sym_equation_dPsi.simplify().simplify()
    sym_equation_dPsi = simplify_eq_with_assumptions(sym.Eq(sym_equation_dPsi, 0)).lhs
    sym_equation_dPsi = sym_equation_dPsi.simplify()

    verbose_print(verbose, "Final equation: ")
    verbose_display(verbose, sym.Eq(sym_equation_dPsi, 0))

    return sym_equation_dPsi, substitutions_dV_to_dPsi


def get_main_equations(verbose: bool = False, super_verbose: bool = False):
    # Getting dPsi equation
    sym_equation_dPsi, substitutions_dV_to_dPsi = get_equation_dPsi(verbose = super_verbose)

    # Create dimensionless dPsi equation
    verbose_display(verbose, 
                    Markdown(r'Dimensionless $\delta{\Psi}$ equation: ')
                    )
    sym_equation_dPsi = substitute(sym_equation_dPsi, substitutions_dimensionless).simplify().simplify()
    sym_d2z_dPsi_dim_expr = sym.solve(sym_equation_dPsi, sym_dPsi_dim.diff(sym_z, 2))[0]
    sym_d2z_dPsi_dim_expr = sym_d2z_dPsi_dim_expr.simplify()
    verbose_display(verbose, sym.Eq(sym_d2z_dPsi_dim_symbol, sym_d2z_dPsi_dim_expr))
    verbose_display(verbose, 
                    Markdown(r'$\delta{\Psi}$ equation in the limit $\epsilon \ll 1$: ')
                    )
    sym_d2z_dPsi_dim_expr = sym_d2z_dPsi_dim_expr.series(sym_epsilon, n = 2).removeO().simplify()
    verbose_display(verbose, sym.Eq(sym_d2z_dPsi_dim_symbol, sym_d2z_dPsi_dim_expr))

    # Creating substitutions dVz -> dPsi according to (55) from POF
    verbose_display(verbose, 
                Markdown(r'Local substitutions: ')
                )
    sym_dPsi_local_subsitution0 = sym_epsilon / sym_mu * sym_dpsi * sym_foo
    sym_mu_local_subsitution = sym.solve(sym.Eq(*substitutions_dimensionless.lambda_mu), sym_mu)[0]
    sym_epsilon_local_subsitution = sym.solve(sym.Eq(*substitutions_dimensionless.O0), sym_epsilon)[0]
    sym_dPsi_local_subsitution1 = sym_dPsi_local_subsitution0.subs({
                    sym_mu: sym_mu_local_subsitution,
                    sym_epsilon: sym_epsilon_local_subsitution
        }).simplify()
    sym_dPsi_local_subsitution = sym_dPsi_local_subsitution1.subs(sym_omega0, sym_omega).simplify()
    sym_dVz_local_subsitution = substitutions_dV_to_dPsi[2][1].subs(sym_dPsi, sym_dPsi_local_subsitution)
    verbose_display(verbose,
        sym.Eq(
            sym_dPsi, 
            sym.Eq(sym_dPsi_local_subsitution0, 
                   sym.Eq(
                       sym_dPsi_local_subsitution1, sym_dPsi_local_subsitution, evaluate = False), 
                   evaluate = False), 
        evaluate = False)
    )
    verbose_display(verbose, 
        sym.Eq(
            sym_dVz_symbol,
            sym.Eq(
                substitutions_dV_to_dPsi[2][1], sym_dVz_local_subsitution, evaluate = False), 
        evaluate = False)                
    )
    substitutions_dimensionless_local = (substitutions_dimensionless.dxi, 
                                         substitutions_dimensionless.dpsi, 
                                         substitutions_dimensionless.psi0)

    # Printing global dimensionless substitutions
    if verbose: print_substitutions(substitutions_dimensionless) 

    # First boundary condition
    verbose_display(verbose, 
                    Markdown(r'First bc $\big($ assume $\dfrac{\lambda}{\Omega^{0}} \ll 1$ $\big)$: ')
                    )

    sym_bc_1 = sym_bc_1_initial
    verbose_display(verbose, sym_bc_1)
    sym_bc_1 = sym_bc_1.subs(sym_dVz_symbol, sym_dVz_local_subsitution).simplify()
    sym_bc_1 = substitute(sym_bc_1, substitutions_dimensionless_local).simplify()
    sym_bc_1 = sym.expand(sym_bc_1 / sym_omega)
    verbose_display(verbose, sym_bc_1)
    sym_bc_1 = sym_bc_1.subs(sym_lambda / sym_omega, sym_zu).series(sym_zu, n = 1).removeO().simplify()
    sym_bc_1 = simplify_eq_with_assumptions(sym.Eq(sym_bc_1, 0)).lhs
    verbose_display(verbose, sym_bc_1)
    
    verbose_print(verbose)

    # Second boundary condition
    verbose_display(verbose, 
                Markdown(r'Second bc $\big($ assume $\dfrac{\lambda}{\Omega^{0}} \ll 1$ $\big)$: ')
                )
    sym_bc_2 = sym_bc_2_initial
    verbose_display(verbose, sym_bc_2)
    sym_bc_2 = sym_bc_2.subs(sym_dp_symbol, sym_dp)
    substitution_omega = sym_omega * (sym_omega + sym_O0)
    verbose_display(super_verbose, sym.Eq(sym_g, substitution_omega))
    sym_bc_2 = sym_bc_2.subs(sym_g, substitution_omega)
    sym_bc_2 = substitute(sym_bc_2, substitutions_dimensionless_local).simplify()
    sym_bc_2 = sym.expand(sym_bc_2 / sym_omega**2)
    verbose_display(verbose, sym_bc_2)
    sym_bc_2 = sym_bc_2.subs(sym_lambda / sym_omega, sym_zu).series(sym_zu, n = 1).removeO().simplify()
    sym_bc_2 = simplify_eq_with_assumptions(sym.Eq(sym_bc_2, 0)).lhs
    verbose_display(verbose, sym_bc_2)

    verbose_print(verbose)

    # mu equation (in terms F(mu))
    verbose_display(verbose, 
                    Markdown(r'$F(\mu)$ equation in the limit $\epsilon \ll 1$ '
                             r'and $\Omega^{(0)} \gg \big(\theta h^{(0)}\big)^2\omega$: ')
                    )
 
    sym_bc_matrix = sym.linear_eq_to_matrix([sym_bc_1, sym_bc_2], [sym_dh_dim, sym_dpsi_dim])[0]
    sym_eq_mu = sym_bc_matrix.det()
    sym_eq_mu = sym_eq_mu.simplify()
    sym_eq_mu = simplify_eq_with_assumptions(sym.Eq(sym_eq_mu, 0)).lhs
    verbose_display(verbose, sym_eq_mu)

    sym_eq_mu_rhs = sym.solve(sym_eq_mu, sym_F_symbol)[0]
    sym_eq_mu = sym_F_symbol - sym_eq_mu_rhs
    verbose_display(verbose, sym_eq_mu)
    sym_eq_mu = substitute(sym_eq_mu, substitutions_dimensionless)
    sym_eq_mu = sym.expand(sym_eq_mu)
    verbose_display(super_verbose, sym_eq_mu)
    sym_eq_mu = sym_eq_mu.simplify()
    sym_eq_mu = simplify_eq_with_assumptions(sym.Eq(sym_eq_mu, 0))
    sym_eq_mu = sym_eq_mu.lhs
    verbose_display(verbose, sym_eq_mu)

    verbose_print(verbose)

    # F(mu) expression
    verbose_display(verbose, 
                Markdown(r'Getting $F(\mu)$: ')
                )
    sym_F_expr = sym_F_expr_initial
    verbose_display(verbose, sym.Eq(sym_F_symbol, sym_F_expr_initial))
    verbose_display(verbose, sym.Eq(sym_dp_int_symbol, sym_dp_int_initial))
    sym_dp_int = substitute(sym_dp_int_initial, substitutions_dV_to_dPsi).simplify()
    sym_F_expr = sym_F_expr.subs(sym_dp_int_symbol, sym_dp_int)
    sym_F_expr = substitute(sym_F_expr, substitutions_dV_to_dPsi).simplify()
    sym_F_expr = substitute(sym_F_expr, substitutions_dimensionless).simplify()
    sym_F_expr = sym_F_expr.simplify()
    sym_dp_int_dim = sym_F_expr.find(sym.Integral).pop()
    sym_F_expr = sym_F_expr.subs(sym_dp_int_dim, sym_dp_int_dim_symbol).simplify()
    verbose_display(verbose, sym.Eq(sym_F_symbol, sym_F_expr))
    verbose_display(verbose, sym.Eq(sym_dp_int_dim_symbol, sym_dp_int_dim))
    verbose_display(verbose, 
                Markdown(r'$F(\mu)$ expression in the limit $\epsilon \ll 1$ '
                         r'and $\Omega^{(0)} \gg \big(\theta h^{(0)}\big)^2\omega$: ')
                )
    sym_F_expr = sym.expand(sym_F_expr)
    verbose_display(super_verbose, sym.Eq(sym_F_symbol, sym_F_expr))
    sym_F_expr = sym_F_expr.series(sym_epsilon, n = 1).removeO()
    verbose_display(super_verbose, sym.Eq(sym_F_symbol, sym_F_expr))
    sym_F_expr = sym_F_expr.subs(sym_theta * sym_h0 / sym_epsilon, sym_zu)
    verbose_display(super_verbose, sym.Eq(sym_F_symbol, sym_F_expr))
    sym_F_expr = sym_F_expr.series(sym_zu, n = 1).removeO()
    sym_F_expr = sym_F_expr.simplify()
    verbose_display(verbose, sym.Eq(sym_F_symbol, sym_F_expr))
    int_expr, int_limits = sym_dp_int_dim.args
    int_expr = int_expr.subs(sym_d2z_dPsi_dim_symbol, sym_d2z_dPsi_dim_expr).simplify()
    sym_dp_int_dim = sym.Integral(int_expr, int_limits)
    verbose_display(verbose, sym.Eq(sym_dp_int_dim_symbol, sym_dp_int_dim))

    verbose_print(verbose)

    verbose_display(verbose, 
                Markdown(r'$\mu$-equation to solve: ')
    )

    sym_eq_mu = sym_eq_mu.subs(sym_F_symbol, sym_F_expr).simplify()
    sym_eq_mu = simplify_eq_with_assumptions(sym.Eq(sym_eq_mu, 0))
    verbose_display(verbose, sym_eq_mu)
    sym_eq_mu = sym_eq_mu.lhs

    return sym_d2z_dPsi_dim_expr, sym_eq_mu, sym_dp_int_dim


def refactor_sym_equation_small_theta(sym_equations: tuple, *, verbose: bool = False) -> tuple:
    sym_d2z_dPsi_dim_expr, sym_eq_mu, sym_dp_int_dim = sym_equations

    sym_eq_mu = sym_eq_mu.series(sym_theta, n = 1).removeO().simplify()
    sym_eq_mu = simplify_eq_with_assumptions(sym.Eq(sym_eq_mu, 0)).lhs

    sym_d2z_dPsi_dim_expr = sym_d2z_dPsi_dim_expr.replace(sym.sqrt(sym_theta**2 + 1), 1)

    expr, limits = sym_dp_int_dim.args
    expr = expr.series(sym_theta, n = 1).removeO().simplify()
    sym_dp_int_dim = sym.Integral(expr, limits)
    
    verbose_display(verbose, sym_d2z_dPsi_dim_expr)
    verbose_display(verbose, sym_eq_mu)
    verbose_display(verbose, sym_dp_int_dim)
    return sym_d2z_dPsi_dim_expr, sym_eq_mu, sym_dp_int_dim


class NumericalParameters(NamedTuple):
    h0: float
    epsilon: float


class NumericalSolution(NamedTuple):
    num_mu: float
    theta: float
    alpha: float
    z: RealNumpyArray
    dPsi: RealNumpyArray
    partial_dPsi0: float


class NumericalProblem:
    global sym_h0, sym_epsilon, sym_theta, sym_mu, sym_dPsi_dim, sym_dp_int_dim_symbol, sym_d2z_dPsi_dim_symbol
    __MAXITER: int = int(1e2)

    def __init__(self, sym_equations: tuple, 
                 numerical_parameters: NumericalParameters, 
                 N_POINTS: int = int(2**15)) -> None:    
        
        if not numerical_parameters.epsilon > numerical_parameters.h0**2:
            raise ValueError('Wrong initial parameters: epsilon must be greater then h0^2.')
        
        self.substitutions_numerical = (
            (sym_h0, numerical_parameters.h0),
            (sym_epsilon, numerical_parameters.epsilon),
            )
        self.__N_POINTS: int = N_POINTS
        self.__sym_equations = sym_equations
        self.__numerical_parameters = numerical_parameters
        self.equation_dPsi, self.num_eq_mu, self.num_dp_int_dim = self._lambdify_sym_equations(sym_equations)

    @property
    def numerical_parameters(self):
        return self.__numerical_parameters

    def show_equations(self) -> None:
        sym_d2z_dPsi_dim_expr, sym_eq_mu, sym_dp_int_dim = self.__sym_equations

        print('Solving the following system: ')
        display(sym.Eq(sym_d2z_dPsi_dim_symbol, sym_d2z_dPsi_dim_expr))
        display(sym.Eq(sym_eq_mu, 0))
        display(sym.Eq(sym_dp_int_dim_symbol, sym_dp_int_dim))

    def _lambdify_sym_equations(self, 
                                sym_equations) -> tuple[Callable[..., float], Callable[..., float], Callable[..., float]]:
        
        sym_d2z_dPsi_dim_expr, sym_eq_mu, sym_dp_int_dim = sym_equations

        # Lambdify equation dPsi
        sym_d2z_dPsi_dim_expr = substitute(sym_d2z_dPsi_dim_expr, self.substitutions_numerical).simplify()
        num_d2z_dPsi_dim = sym.lambdify([sym_mu, 
                                         sym_theta,
                                         sym_alpha,
                                         sym_z, 
                                         sym_dPsi_dim], 
                                         sym_d2z_dPsi_dim_expr, modules = 'numpy')
        
        equation_dPsi = lambda num_mu, theta, alpha, z, y: np.array((y[1], num_d2z_dPsi_dim(num_mu, theta, alpha, z, y[0])))
        
        # Lambdify dp_dim integral
        sym_dp_int_dim_expr = sym_dp_int_dim.args[0]
        sym_dp_int_dim_expr = substitute(sym_dp_int_dim_expr, self.substitutions_numerical)
        num_dp_int_dim_expr = sym.lambdify([sym_mu,
                                            sym_theta,
                                            sym_alpha, 
                                            sym_z, 
                                            sym_dPsi_dim], 
                                            sym_dp_int_dim_expr, modules = 'numpy')

        num_dp_int_dim = lambda num_mu, theta, alpha, z, dPsi: simpson(y = num_dp_int_dim_expr(num_mu, theta, alpha, z, dPsi), x = -z)

        # Lambdify base mu equation
        sym_eq_mu = substitute(sym_eq_mu, self.substitutions_numerical).simplify()
        sym_eq_mu = sym_eq_mu.subs(sym_dPsi_dim.diff(sym_z), sym_zu)
        num_eq_mu_lmd = sym.lambdify([sym_mu,
                                  sym_theta,
                                  sym_alpha,
                                  sym_zu,
                                  sym_dp_int_dim_symbol],
                                  sym_eq_mu, modules = 'numpy')
        
        num_eq_mu = lambda num_mu, theta, alpha, partial_dPsi0, num_dp_int: num_eq_mu_lmd(num_mu, theta, alpha, partial_dPsi0, num_dp_int)

        return equation_dPsi, num_eq_mu, num_dp_int_dim  

    def mu_problem(self, 
                   x: float, 
                   theta: float, 
                   alpha: float, 
                   partial_dPsi0_initial: float) -> float:
        
        # print(f'\tCalling mu equation with mu = {x:.3e}')
        z, dPsi, partial_dPsi0 = self.solve_dPsi_eq(x, theta, alpha, partial_dPsi0_initial)
        dp_int = self.num_dp_int_dim(x, theta, alpha, z, dPsi)
        return self.num_eq_mu(x, theta, alpha, partial_dPsi0, dp_int)

    def estimate_partial_dPsi0(self, 
                               num_mu: float, 
                               theta: float, 
                               alpha: float, 
                               CL2: bool = False) -> float:
        substitutions_numerical_local = (
            (sym_theta, theta),
            (sym_alpha, alpha),
            (sym_mu, num_mu),
            (sym_z, 0)
        )  

        if CL2:
            sym_dPsi_dim_CL2 = sym.besseli(sym_theta, sym_mu * sym.exp(sym_z))
            tmp = substitute(sym_dPsi_dim_CL2.diff(sym_z).simplify(), self.substitutions_numerical)
        else:
            global sym_dPsi_dim_est
            tmp = substitute(sym_dPsi_dim_est.diff(sym_z).simplify(), self.substitutions_numerical)
        
        partial_dPsi0_est = substitute(tmp, substitutions_numerical_local)
        return float(partial_dPsi0_est)
    
    def _shooting_method(self, 
                         num_mu: float, 
                         partial_dPsi0: float, 
                         equation_dPsi_metadata) -> tuple[RealNumpyArray, RealNumpyArray]:
        
        z, equation_dPsi_to_solve = equation_dPsi_metadata
        z_span = (z[0], z[-1])
        dPsi0, partial_dPsi0 = 0, partial_dPsi0
        y0 = [dPsi0, partial_dPsi0]
        solution = solve_ivp(lambda z, y: equation_dPsi_to_solve(num_mu, z, y), 
                             z_span, y0, t_eval = z, 
                             method = 'RK45', 
                             atol = 1e-18, 
                             rtol = 1e-13)
        dPsi = solution.y[0]
        return z, dPsi
    
    def _calculate_partial_dPsi0(self, 
                                 num_mu: float, 
                                 partial_dPsi0_initial: float, 
                                 equation_dPsi_metadata) -> float:
        
        partial_dPsi0_problem = lambda x: self._shooting_method(num_mu, x, equation_dPsi_metadata)[1][-1]
        partial_dPsi0 = newton(func = partial_dPsi0_problem, 
                                x0 = partial_dPsi0_initial,
                                tol = 1e-14,
                                maxiter = self.__MAXITER)
        return partial_dPsi0
    
    def _create_equation_dPsi_metadata(self, 
                                       theta: float,
                                       alpha: float) -> tuple[RealNumpyArray, Callable]:
        
        equation_dPsi_to_solve = lambda num_mu, z, dPsi: self.equation_dPsi(num_mu, theta, alpha, z, dPsi)
        MAX_z = -3 / theta
        z = np.linspace(0, MAX_z, self.__N_POINTS, dtype = float)
        return z, equation_dPsi_to_solve

    def solve_dPsi_eq(self, 
                      num_mu: float, 
                      theta: float, 
                      alpha: float, 
                      partial_dPsi0_initial: float) -> tuple[RealNumpyArray, RealNumpyArray, float]:

        equation_dPsi_metadata = self._create_equation_dPsi_metadata(theta, alpha)

        partial_dPsi0 = self._calculate_partial_dPsi0(num_mu, partial_dPsi0_initial, equation_dPsi_metadata)
        z, dPsi = self._shooting_method(num_mu, partial_dPsi0, equation_dPsi_metadata)
        return z, dPsi, partial_dPsi0
    
    def solve_mu_eq_newton(self, 
                           num_mu_initial: float,
                           theta: float,
                           alpha: float,
                           partial_dPsi0_initial: float | None = None) -> NumericalSolution:
        
        if not partial_dPsi0_initial:
            partial_dPsi0_initial = self.estimate_partial_dPsi0(num_mu_initial, theta, alpha)
        
        num_mu = newton(func = self.mu_problem, 
                        x0 = num_mu_initial, 
                        args = (theta, alpha, partial_dPsi0_initial),
                        tol = 1e-14,
                        maxiter = self.__MAXITER)
        z, dPsi, partial_dPsi0 = self.solve_dPsi_eq(num_mu, theta, alpha, partial_dPsi0_initial)
        return NumericalSolution(num_mu, theta, alpha, z, dPsi, partial_dPsi0)  
    
    @classmethod
    def calculate_fields(cls, 
                         num_mu: float, 
                         theta: float, 
                         z: RealNumpyArray, 
                         dPsi: RealNumpyArray) -> tuple[RealNumpyArray, RealNumpyArray, RealNumpyArray]:
        
        partial_z = lambda field, z: np.diff(field) / np.diff(z)

        dVy = partial_z(dPsi, z)
        dVz = theta * dPsi
        dOx = num_mu**2 * np.exp(2 * z) * dPsi - np.exp(2 * z)

        return dVy, dVz, dOx


def save_dump(sym_equations: tuple, numerical_parameters: NumericalParameters, dump_filename: str) -> None:
    numerical_problem = NumericalProblem(sym_equations, numerical_parameters)
    with open(dump_filename, 'wb') as dump_fp:
        dill.dump(numerical_problem, dump_fp, recurse = True)


def load_dump(dump_filename: str) -> NumericalProblem:
    with open(dump_filename, 'rb') as dump_fp:
        numerical_problem = dill.load(dump_fp)
    return numerical_problem


def create_experiment(h0: float, epsilon: float) -> None:
    sym_equations = get_main_equations(verbose = False, super_verbose = False)
    numerical_parameters = NumericalParameters(h0 = h0, epsilon = epsilon)
    dump_filename = f"numerical_problem_ho_{h0:.2e}_epsilon_{epsilon:.2e}.pkl"
    save_dump(sym_equations, numerical_parameters, dump_filename)
    dump_file_path = Path(dump_filename).absolute().as_posix()
    print(f"Experiment was successfully created: {dump_file_path}")
    