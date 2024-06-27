import sympy as sym
from IPython.display import display, Markdown
from collections import namedtuple

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

class module_POF:   

    def __init__(self, main_wave: bool, oblique_wave: bool, shear_flow: bool):
        self.__if_main_wave = main_wave
        self.__if_oblique_wave = oblique_wave
        self.__if_shear_flow = shear_flow

        self.sym_g = sym.Symbol('g', real = True, positive = True, nonzero = True, zero = False)
        self.sum_nu = sym.Symbol(r'\nu', real = True, positive = True, nonzero = True, zero = False)

        # 1) Main wave (initially set to zero)
        self.sym_phi0_symbol = sym.Symbol(r'\phi^{\scriptsize{(0)}}')
        self.sym_omega0 = sym.Symbol(r'\omega_0', real = True, positive = True, nonzero = True, zero = False)
        self.sym_omega = sym.Symbol(r'\omega', real = True, positive = True, nonzero = True, zero = False)
        self.sym_psi0, self.sym_h0, self.sym_phi0 = self.__initialize_main_wave()

        # 4) Interaction parameters
        self.sym_lambda = sym.Symbol(r'\lambda', zero = False, nonzero = True)        
        self.sym_mu = sym.Symbol(r'\mu', zero = False, nonzero = True)
        self.sym_mu0 = sym.Symbol(r'\mu_0', zero = False, nonzero = True)

        # 2) Oblique wave (initially set to zero)
        self.sym_dphi_symbol = sym.Symbol(r'\delta{\phi}')
        self.sym_theta = sym.Symbol(r'\theta', real = True, positive = True, nonzero = True, zero = False)
        self.sym_dkx, self.sym_dky, self.sym_dkz = 1, self.sym_theta, sym.sqrt(1 + self.sym_theta**2)
        self.sym_dpsi, self.sym_dxi, self.sym_dphi = self.__initialize_oblique_wave()
        self.sym_dh_dim = sym.Symbol(r'\tilde{\delta h}', nonzero = True, zero = False)
        self.sym_dpsi_dim = sym.Symbol(r'\tilde{\delta\psi}', nonzero = True, zero = False)

        # 3) Initial shear flow (initially set to zero)
        self.sym_V0_symbol = sym.Symbol(r'V^{\scriptsize{(0)}}', real = True, positive = False)
        self.sym_vec_O0, self.sym_vec_V0 = self.__initialize_shear_flow()
        self.sym_O0x, self.sym_O0y, self.sym_O0z = self.sym_vec_O0
        self.sym_O0 = self.sym_O0y 
        self.sym_V0x, self.sym_V0y, self.sym_V0z = self.sym_vec_V0
        self.sym_V0 = self.sym_V0x 
        self.sym_epsilon = sym.Symbol(r'\epsilon', real = True, positive = True, zero = False, nonzero = True)

        # 5) Perturbated flow 
        self.sym_dVx_symbol = sym.Function(r'\delta{V}_x')(sym_z)
        self.sym_dVy_symbol = sym.Function(r'\delta{V}_y')(sym_z)
        self.sym_dVz_symbol = sym.Function(r'\delta{V}_z')(sym_z)
        self.sym_dVx = self.sym_dVx_symbol * sym.exp(sym.I * self.sym_theta * sym_y) * sym.exp(self.sym_lambda * sym_t)
        self.sym_dVy = self.sym_dVy_symbol * sym.exp(sym.I * self.sym_theta * sym_y)* sym.exp(self.sym_lambda * sym_t)
        self.sym_dVz = self.sym_dVz_symbol * sym.exp(sym.I * self.sym_theta * sym_y)* sym.exp(self.sym_lambda * sym_t)

        self.sym_dOx_symbol = sym.Function(r'\delta{\Omega}_x')(sym_z)
        self.sym_dOy_symbol = sym.Function(r'\delta{\Omega}_y')(sym_z)
        self.sym_dOz_symbol = sym.Function(r'\delta{\Omega}_z')(sym_z)
        self.sym_vec_dO = curl((self.sym_dVx, self.sym_dVy, self.sym_dVz))
        self.sym_dOx, self.sym_dOy, self.sym_dOz = self.sym_vec_dO

        self.sym_dPsi = sym.Function(r'\delta\Psi')(sym_z)
        self.sym_dPsi_symbol = self.sym_dPsi
        self.sym_d2z_dPsi = self.sym_dPsi_symbol.diff((sym_z, 2))
        self.sym_dPsi_dim = sym.Function(r'\tilde{\delta\Psi}')(sym_z)
        self.sym_d2z_dPsi_dim_symbol = (self.sym_dPsi_dim).diff(sym_z, 2)
        self.sym_dPsi_dim_mu = sym.Function(r'\delta\Psi')(self.sym_mu)
        self.sym_partial_dPsi_dim_mu = sym.Function(r'\delta\Psi^{\prime}')(self.sym_mu)

        # 6) Perturbated vortex force
        self.sym_dfVx_symbol = sym.Symbol(r'\delta f^{\scriptsize{V}}_x')
        self.sym_dfVy_symbol = sym.Symbol(r'\delta f^{\scriptsize{V}}_y')
        self.sym_dfVz_symbol = sym.Symbol(r'\delta f^{\scriptsize{V}}_z')
        self.sym_curl_dfVx_symbol = sym.Symbol(r'\big(\operatorname{curl}\delta f^{\scriptsize{V}}\big)_x')
        self.sym_curl_dfVy_symbol = sym.Symbol(r'\big(\operatorname{curl}\delta f^{\scriptsize{V}}\big)_y')
        self.sym_curl_dfVz_symbol = sym.Symbol(r'\big(\operatorname{curl}\delta f^{\scriptsize{V}}\big)_z')

        # 7) Vortical pressure part
        self.sym_dp_symbol = sym.Symbol(r'\delta{p}^{\varpi}')
        self.sym_dp_int_symbol = sym.Symbol(r'\delta{p}^{\varpi}_{int}')
        self.sym_dp_int_symbol_mu = sym.Function(r'\delta{p}^{\varpi}_{int}')(self.sym_mu)
        self.sym_dp_int_dim_symbol = sym.Symbol(r'\overline{\delta{p}}^{\varpi}_{int}')
        self.sym_dp_int_initial = sym.Integral((-1) * sym.exp((self.sym_dkz + 1) * sym_z) * \
                                    ((sym.I * self.sym_dVx_symbol.diff((sym_z, 2)) - sym.I * self.sym_theta**2 * self.sym_dVx_symbol) + \
                                    (self.sym_dVz_symbol.diff((sym_z, 2)) - self.sym_theta**2 * self.sym_dVz_symbol)), \
                                    (sym_z, -sym.oo, 0)).simplify()
        self.sym_F_symbol = sym.Function(r'F')(self.sym_mu)
        self.sym_dp = sym.I * self.sym_O0 * self.sym_dpsi * self.sym_F_symbol
        self.sym_F_expr_initial = (sym_alpha * self.sym_psi0 * self.sym_dp_int_symbol / (sym.I * self.sym_O0 * self.sym_dpsi) + 1 + 
                           sym_alpha * sym.I * self.sym_psi0 * self.sym_dVx_symbol.diff(sym_z) / (sym.I * self.sym_O0 * self.sym_dpsi)) / self.sym_dkz
        # self.sym_dp = (sym_alpha * self.sym_psi0 * self.sym_dp_int_symbol + sym.I * self.sym_O0 * self.sym_dpsi + 
        #                sym_alpha * sym.I * self.sym_psi0 * self.sym_dVx_symbol.diff(sym_z)) / self.sym_dkz

        # 8) Boundary conditions
        # equation (46)
        self.sym_bc_1_initial = (self.sym_lambda - sym.I * self.sym_omega) * self.sym_dxi - self.sym_dkz * self.sym_dpsi - \
                                                                            sym_alpha * self.sym_h0 * self.sym_dVz_symbol.diff(sym_z)
        # equation (47)
        self.sym_bc_2_initial = (self.sym_lambda - sym.I * self.sym_omega) * self.sym_dpsi + self.sym_g * self.sym_dxi - self.sym_dp_symbol
        
        # Dimensionless parameters
        self.substitutions_dimensionless = self.__initialize_substitutions_dimensionless()

        # Analytical expression dPsi (55) from POF
        sym_j0mu_frac = (1 - sym.besselj(0, self.sym_mu)) / sym.besselj(0, self.sym_mu)
        self.sym_dPsi_dim_est = (1 / self.sym_mu**2 *  (1 - (1 + sym_alpha * sym_j0mu_frac * sym_zeta**self.sym_theta) * sym.besselj(0, self.sym_mu * sym_zeta)))

    def __initialize_substitutions_dimensionless(self):
        if self.__if_main_wave and self.__if_shear_flow:
            self.sym_lambda_subs = sym.sqrt(2 * self.sym_O0 * self.sym_omega) * self.sym_h0 * self.sym_theta / self.sym_mu 
            self.sym_dPsi_subs = self.sym_dpsi * self.sym_mu * self.sym_epsilon * self.sym_dPsi_dim / sym_alpha
            # self.sym_dPsi_subs = self.sym_dpsi * self.sym_mu * self.sym_dPsi_dim
            # self.sym_dPsi_subs = self.sym_dpsi * self.sym_dPsi_dim
            DimensionlessSubstitutions = namedtuple('DimensionlessSubstitutions', ['lambda_mu', 'dPsi', 'dxi', 'dpsi', 'psi0', 'omega', 'O0', 'g'])
            substitutions_dimensionless = DimensionlessSubstitutions(
                (self.sym_lambda, self.sym_lambda_subs), 
                (self.sym_dPsi, self.sym_dPsi_subs), 
                (self.sym_dxi, self.sym_dh_dim * self.sym_h0), 
                (self.sym_dpsi, self.sym_dpsi_dim * self.sym_psi0), 
                (self.sym_psi0, -sym.I * self.sym_omega * self.sym_h0),
                (self.sym_omega, self.sym_omega0 - self.sym_O0 / 2),
                (self.sym_O0, self.sym_omega0 * self.sym_epsilon**2 / 2), 
                (self.sym_g, self.sym_omega0**2)         
            )

        else: substitutions_dimensionless = ()
        return substitutions_dimensionless
    
    def __initialize_main_wave(self):
        if self.__if_main_wave: 
            sym_psi0 = sym.Symbol(r'\psi^{\scriptsize{(0)}}', real = False)
            sym_h0 = sym.Symbol(r'h^{\scriptsize{(0)}}', real = True, positive = True, nonzero = True, zero = False)
            sym_phi0 = sym_psi0 * sym.exp(sym.I * sym_x + sym_z - sym.I * self.sym_omega * sym_t)
        else: sym_psi0 = sym_h0 = sym_phi0 = sym.S.Zero

        return sym_psi0, sym_h0, sym_phi0
    
    def __initialize_oblique_wave(self):
        if self.__if_oblique_wave: 
            sym_dpsi = sym.Symbol(r'\delta\psi', real = False)
            sym_dxi = sym.Symbol(r'\delta\xi', zero = False, nonzero = True)
            sym_dphi = sym_dpsi * sym.exp(sym.I * self.sym_dkx * sym_x + sym.I * self.sym_dky * sym_y \
                                + self.sym_dkz * sym_z - sym.I * self.sym_omega * sym_t) \
                                * sym.exp(self.sym_lambda * sym_t)
        else: sym_dpsi = sym_dxi = sym_dphi = sym.S.Zero


        return sym_dpsi, sym_dxi, sym_dphi
    
    def __initialize_shear_flow(self):
        if self.__if_shear_flow: 
            sym_O0 = sym.Symbol(r'\Omega^{\scriptsize{(0)}}', real = True, positive = True, zero = False, nonzero = True)
            sym_V0 = sym_O0 * sym_z
            sym_vec_O0 = (sym.S.Zero, sym_O0, sym.S.Zero)
            sym_vec_V0 = (sym_V0, sym.S.Zero, sym.S.Zero)
        else:
            sym_O0 = sym_V0 = sym.S.Zero
            sym_vec_O0 = sym_vec_V0 = sym_zero_vec

        return sym_vec_O0, sym_vec_V0


def curl(vec: tuple):
    vec_x, vec_y, vec_z = vec
    return (vec_z.diff(sym_y) - vec_y.diff(sym_z), \
            vec_x.diff(sym_z) - vec_z.diff(sym_x), \
            vec_y.diff(sym_x) - vec_x.diff(sym_y))


def verbose_print(verbose: bool, *objs):
    return print(*objs) if verbose else None


def verbose_display(verbose: bool, *objs):
    return display(*objs) if verbose else None


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
    assert isinstance(vec, tuple)
    return tuple(map(lambda element: element.simplify(), vec))


def small_parameter(expr, variable, n = 2):
    return expr.series(variable, x0 = 0, n = n).removeO() 


def simplify_eq_with_assumptions(eq):
    try:
        assert eq.rhs == 0  # assert that right-hand side is zero
        assert type(eq.lhs) == sym.Mul  # assert that left-hand side is a multipl.
        newargs = []  # define a list of new multiplication factors.
        for arg in eq.lhs.args:
            if arg.is_nonzero:
                continue  # arg is positive, let's skip it.
            newargs.append(arg)
        # rebuild the equality with the new arguments:
        return sym.Eq(eq.lhs.func(*newargs), 0)
    except:
        return eq
