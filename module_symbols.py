import sympy as sym
from module_utils import *
from collections import namedtuple


def __initialize_substitutions_dimensionless():
    sym_lambda_subs = sym.sqrt(2 * sym_O0 * sym_omega) * sym_h0 * sym_theta / sym_mu 
    sym_dPsi_subs = sym_dpsi * sym_mu * sym_epsilon * sym_dPsi_dim / sym_alpha
    DimensionlessSubstitutions = namedtuple('DimensionlessSubstitutions', ['lambda_mu', 'dPsi', 'dxi', 'dpsi', 'psi0', 'omega', 'O0', 'g'])
    substitutions_dimensionless = DimensionlessSubstitutions(
        (sym_lambda, sym_lambda_subs), 
        (sym_dPsi, sym_dPsi_subs), 
        (sym_dxi, sym_dh_dim * sym_h0), 
        (sym_dpsi, sym_dpsi_dim * sym_psi0), 
        (sym_psi0, -sym.I * sym_omega * sym_h0),
        (sym_omega, sym_omega0 - sym_O0 / 2),
        (sym_O0, sym_omega0 * sym_epsilon**2 / 2), 
        (sym_g, sym_omega0**2)         
        )
    return substitutions_dimensionless


def __initialize_main_wave():
    sym_psi0 = sym.Symbol(r'\psi^{\scriptsize{(0)}}', real = False)
    sym_h0 = sym.Symbol(r'h^{\scriptsize{(0)}}', real = True, positive = True, nonzero = True, zero = False)
    sym_phi0 = sym_psi0 * sym.exp(sym.I * sym_x + sym_z - sym.I * sym_omega * sym_t)
    return sym_psi0, sym_h0, sym_phi0


def __initialize_oblique_wave():
    sym_dpsi = sym.Symbol(r'\delta\psi', real = False)
    sym_dxi = sym.Symbol(r'\delta\xi', zero = False, nonzero = True)
    sym_dphi = sym_dpsi * sym.exp(sym.I * sym_dkx * sym_x + sym.I * sym_dky * sym_y \
                        + sym_dkz * sym_z - sym.I * sym_omega * sym_t) \
                        * sym.exp(sym_lambda * sym_t)
    return sym_dpsi, sym_dxi, sym_dphi


def __initialize_shear_flow():
    sym_O0 = sym.Symbol(r'\Omega^{\scriptsize{(0)}}', real = True, positive = True, zero = False, nonzero = True)
    sym_V0 = sym_O0 * sym_z
    sym_vec_O0 = (sym.S.Zero, sym_O0, sym.S.Zero)
    sym_vec_V0 = (sym_V0, sym.S.Zero, sym.S.Zero)
    return sym_vec_O0, sym_vec_V0


def curl(vec: tuple):
    vec_x, vec_y, vec_z = vec
    return (vec_z.diff(sym_y) - vec_y.diff(sym_z), \
            vec_x.diff(sym_z) - vec_z.diff(sym_x), \
            vec_y.diff(sym_x) - vec_x.diff(sym_y))


def print_substitutions(substitutions: tuple):
    print("Substitutions: ")
    for initial, final in substitutions:
        display(sym.Eq(initial, final, evaluate = False))
    

def substitute(expr, substitutions: tuple):
    if not substitutions: return expr
    for initial, final in substitutions:
        expr = expr.subs(initial, final)
    return expr


def vector_mult(vec1, vec2):
    vec1_x, vec1_y, vec1_z = vec1
    vec2_x, vec2_y, vec2_z = vec2
    return (vec1_y * vec2_z - vec1_z * vec2_y, \
        - vec1_x * vec2_z + vec1_z * vec2_x, \
            vec1_x * vec2_y - vec1_y * vec2_x)


def simplify_vector(vec: tuple) -> tuple:
    return tuple(map(lambda element: element.simplify(), vec))


def small_parameter(expr, variable, n = 2):
    return expr.series(variable, x0 = 0, n = n).removeO() 


def simplify_eq_with_assumptions(eq):
    try:
        assert eq.rhs == 0  
        assert type(eq.lhs) == sym.Mul  
        newargs = []  
        for arg in eq.lhs.args:
            if arg.is_nonzero:
                continue 
            newargs.append(arg)
        return sym.Eq(eq.lhs.func(*newargs), 0)
    except:
        return eq


sym_zero_vec = (sym.S.Zero, sym.S.Zero, sym.S.Zero)
sym_zu = sym.Symbol(r'\zeta', real = True, positive = True, zero = False, nonzero = True)

sym_alpha = sym.Symbol(r'\alpha', real = True, positive = True, zero = False, nonzero = True)

sym_x = sym.Symbol('x', real = True)
sym_y = sym.Symbol('y', real = True)
sym_z = sym.Symbol('z', real = True, positive = False)
sym_t = sym.Symbol('t', real = True, negative = False)
sym_zeta = sym.exp(sym_z)

sym_gamma1 = sym.Symbol(r'\gamma_1', real = True, positive = True, zero = False, nonzero = True)
sym_gamma2 = sym.Symbol(r'\gamma_2', real = True, positive = True, zero = False, nonzero = True)

sym_foo = sym.Function(r'f')(sym_z)

sym_g = sym.Symbol('g', real = True, positive = True, nonzero = True, zero = False)
sum_nu = sym.Symbol(r'\nu', real = True, positive = True, nonzero = True, zero = False)

# 1) Main wave (initially set to zero)
sym_phi0_symbol = sym.Symbol(r'\phi^{\scriptsize{(0)}}')
sym_omega0 = sym.Symbol(r'\omega_0', real = True, positive = True, nonzero = True, zero = False)
sym_omega = sym.Symbol(r'\omega', real = True, positive = True, nonzero = True, zero = False)
sym_psi0, sym_h0, sym_phi0 = __initialize_main_wave()

# 4) Interaction parameters
sym_lambda = sym.Symbol(r'\lambda', zero = False, nonzero = True)        
sym_mu = sym.Symbol(r'\mu', zero = False, nonzero = True)
sym_mu0 = sym.Symbol(r'\mu_0', zero = False, nonzero = True)

# 2) Oblique wave (initially set to zero)
sym_dphi_symbol = sym.Symbol(r'\delta{\phi}')
sym_theta = sym.Symbol(r'\theta', real = True, positive = True, nonzero = True, zero = False)
sym_dkx, sym_dky, sym_dkz = 1, sym_theta, sym.sqrt(1 + sym_theta**2)
sym_dpsi, sym_dxi, sym_dphi = __initialize_oblique_wave()
sym_dh_dim = sym.Symbol(r'\tilde{\delta h}', nonzero = True, zero = False)
sym_dpsi_dim = sym.Symbol(r'\tilde{\delta\psi}', nonzero = True, zero = False)

# 3) Initial shear flow (initially set to zero)
sym_V0_symbol = sym.Symbol(r'V^{\scriptsize{(0)}}', real = True, positive = False)
sym_vec_O0, sym_vec_V0 = __initialize_shear_flow()
sym_O0x, sym_O0y, sym_O0z = sym_vec_O0
sym_O0 = sym_O0y 
sym_V0x, sym_V0y, sym_V0z = sym_vec_V0
sym_V0 = sym_V0x 
sym_epsilon = sym.Symbol(r'\epsilon', real = True, positive = True, zero = False, nonzero = True)

# 5) Perturbated flow 
sym_dVx_symbol = sym.Function(r'\delta{V}_x')(sym_z)
sym_dVy_symbol = sym.Function(r'\delta{V}_y')(sym_z)
sym_dVz_symbol = sym.Function(r'\delta{V}_z')(sym_z)
sym_dVx = sym_dVx_symbol * sym.exp(sym.I * sym_theta * sym_y) * sym.exp(sym_lambda * sym_t)
sym_dVy = sym_dVy_symbol * sym.exp(sym.I * sym_theta * sym_y)* sym.exp(sym_lambda * sym_t)
sym_dVz = sym_dVz_symbol * sym.exp(sym.I * sym_theta * sym_y)* sym.exp(sym_lambda * sym_t)

sym_dOx_symbol = sym.Function(r'\delta{\Omega}_x')(sym_z)
sym_dOy_symbol = sym.Function(r'\delta{\Omega}_y')(sym_z)
sym_dOz_symbol = sym.Function(r'\delta{\Omega}_z')(sym_z)
sym_vec_dO = curl((sym_dVx, sym_dVy, sym_dVz))
sym_dOx, sym_dOy, sym_dOz = sym_vec_dO

sym_dPsi = sym.Function(r'\delta\Psi')(sym_z)
sym_dPsi_symbol = sym_dPsi
sym_d2z_dPsi = sym_dPsi_symbol.diff((sym_z, 2))
sym_dPsi_dim = sym.Function(r'\tilde{\delta\Psi}')(sym_z)
sym_d2z_dPsi_dim_symbol = (sym_dPsi_dim).diff(sym_z, 2)
sym_dPsi_dim_mu = sym.Function(r'\delta\Psi')(sym_mu)
sym_partial_dPsi_dim_mu = sym.Function(r'\delta\Psi^{\prime}')(sym_mu)

# 6) Perturbated vortex force
sym_dfVx_symbol = sym.Symbol(r'\delta f^{\scriptsize{V}}_x')
sym_dfVy_symbol = sym.Symbol(r'\delta f^{\scriptsize{V}}_y')
sym_dfVz_symbol = sym.Symbol(r'\delta f^{\scriptsize{V}}_z')
sym_curl_dfVx_symbol = sym.Symbol(r'\big(\operatorname{curl}\delta f^{\scriptsize{V}}\big)_x')
sym_curl_dfVy_symbol = sym.Symbol(r'\big(\operatorname{curl}\delta f^{\scriptsize{V}}\big)_y')
sym_curl_dfVz_symbol = sym.Symbol(r'\big(\operatorname{curl}\delta f^{\scriptsize{V}}\big)_z')

# 7) Vortical pressure part
sym_dp_symbol = sym.Symbol(r'\delta{p}^{\varpi}')
sym_dp_int_symbol = sym.Symbol(r'\delta{p}^{\varpi}_{int}')
sym_dp_int_symbol_mu = sym.Function(r'\delta{p}^{\varpi}_{int}')(sym_mu)
sym_dp_int_dim_symbol = sym.Symbol(r'\overline{\delta{p}}^{\varpi}_{int}')
sym_dp_int_initial = sym.Integral((-1) * sym.exp((sym_dkz + 1) * sym_z) * \
                            ((sym.I * sym_dVx_symbol.diff((sym_z, 2)) - sym.I * sym_theta**2 * sym_dVx_symbol) + \
                            (sym_dVz_symbol.diff((sym_z, 2)) - sym_theta**2 * sym_dVz_symbol)), \
                            (sym_z, -sym.oo, 0)).simplify()
sym_F_symbol = sym.Function(r'F')(sym_mu)
sym_dp = sym.I * sym_O0 * sym_dpsi * sym_F_symbol
sym_F_expr_initial = (sym_alpha * sym_psi0 * sym_dp_int_symbol / (sym.I * sym_O0 * sym_dpsi) + 1 + 
                    sym_alpha * sym.I * sym_psi0 * sym_dVx_symbol.diff(sym_z) / (sym.I * sym_O0 * sym_dpsi)) / sym_dkz
# sym_dp = (sym_alpha * sym_psi0 * sym_dp_int_symbol + sym.I * sym_O0 * sym_dpsi + 
#                sym_alpha * sym.I * sym_psi0 * sym_dVx_symbol.diff(sym_z)) / sym_dkz

# 8) Boundary conditions
# equation (46)
sym_bc_1_initial = (sym_lambda - sym.I * sym_omega) * sym_dxi - sym_dkz * sym_dpsi - \
                                                                    sym_alpha * sym_h0 * sym_dVz_symbol.diff(sym_z)
# equation (47)
sym_bc_2_initial = (sym_lambda - sym.I * sym_omega) * sym_dpsi + sym_g * sym_dxi - sym_dp_symbol

# Dimensionless parameters
substitutions_dimensionless = __initialize_substitutions_dimensionless()

# Analytical expression dPsi (55) from POF
sym_j0mu_frac = (1 - sym.besselj(0, sym_mu)) / sym.besselj(0, sym_mu)
sym_dPsi_dim_est = (1 / sym_mu**2 *  (1 - (1 + sym_alpha * sym_j0mu_frac * sym_zeta**sym_theta) * sym.besselj(0, sym_mu * sym_zeta)))


