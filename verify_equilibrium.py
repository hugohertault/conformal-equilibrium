"""Symbolic verification of the equilibrium results of
'The conformal mode as an entropic equilibrium' (Secs. 2-5).

Checks:
  [1] Stationary point of F[sigma] and its uniqueness / convexity   (Prop. 1)
  [2] Bekenstein normalisation <=> holographic vacuum condition     (Prop. 2)
  [3] Rigidity: only the exponent difference a-b is fixed           (Sec. 3.1)
  [4] Kinetic coefficient (D-1)(D-2) and canonical normalisation    (Sec. 5)
  [5] Equilibrium correction c_d^eq = -(d+1)/(2(d+2))               (Eq. 12)
      and the decomposition c_d = c_d^eq + (d-1)/(3d(d+2))          (Remark 4)
"""
import sympy as sp

ok = lambda name: print(f"  [PASS] {name}")

# ----------------------------------------------------------------------
# [1] Equilibrium (Proposition 1)
# ----------------------------------------------------------------------
sigma, S, R, V, rho, D = sp.symbols('sigma S R V rho D', positive=True)
F = rho*V*sp.exp((D-1)*sigma) + S/(2*sp.pi*R)*sp.exp(-sigma)

sol = sp.solve(sp.diff(F, sigma), sigma)
assert len(sol) == 1, "stationary point must be unique"
sigma_star = sol[0]
lhs = sp.exp(D*sigma_star)
rhs = S/(2*sp.pi*R*V*rho*(D-1))
assert sp.simplify(lhs - rhs) == 0
ok("e^{D sigma*} = S_ent / (2 pi R V rho (D-1))")

d2_generic = sp.simplify(sp.diff(F, sigma, 2))
assert sp.simplify(d2_generic - ((D-1)**2*rho*V*sp.exp((D-1)*sigma)
                                 + S/(2*sp.pi*R)*sp.exp(-sigma))) == 0
# both terms are manifestly positive for D > 1, positive parameters
assert sp.ask(sp.Q.positive(d2_generic),
              sp.Q.positive(D - 1)) is True
ok("F''(sigma) > 0 for D > 1: the extremum is a minimum")

# ----------------------------------------------------------------------
# [2] Bekenstein normalisation (Proposition 2), D = 4, d = 3
# ----------------------------------------------------------------------
G = sp.symbols('G', positive=True)
Rh = sp.symbols('Rhat', positive=True)
Vh = sp.Rational(4, 3)*sp.pi*Rh**3
A = 4*sp.pi*Rh**2

denominator = 2*sp.pi*Rh*Vh*rho*3          # (D-1) = 3
S_bek = A/(4*G)

rho_sol = sp.solve(sp.Eq(denominator, S_bek), rho)
assert len(rho_sol) == 1
assert sp.simplify(rho_sol[0] - 1/(8*sp.pi*G*Rh**2)) == 0
ok("denominator = A/4G  <=>  rho_L = 1/(8 pi G Rhat^2)   (iff, both ways)")

# forward direction explicitly
assert sp.simplify(denominator.subs(rho, 1/(8*sp.pi*G*Rh**2)) - S_bek) == 0
ok("under the holographic condition the denominator is exactly A/4G")

# ----------------------------------------------------------------------
# [3] Rigidity of the exponents (Sec. 3.1)
# ----------------------------------------------------------------------
a, b, C1, C2 = sp.symbols('a b C1 C2', positive=True)
Fg = C1*sp.exp(a*sigma) - C2*sp.exp(-b*sigma)     # signs: generic two-term
# stationary point of C1 e^{a s} + C2 e^{b s} with b<0: use b -> -b form
Fg = C1*sp.exp(a*sigma) + C2*sp.exp(-b*sigma)
st = sp.solve(sp.diff(Fg, sigma), sigma)[0]
assert sp.simplify(sp.exp((a+b)*st) - C2*b/(C1*a)) == 0
ok("stationarity fixes only e^{(a-b) sigma*}: exponent difference a-b = D")

# ----------------------------------------------------------------------
# [4] Kinetic coefficient and canonical normalisation (Sec. 5)
# ----------------------------------------------------------------------
# sqrt(-g) R = e^{(D-2)sigma} [ -2(D-1) box(sigma) - (D-1)(D-2) (d sigma)^2 ]
# on a flat reference; integrating the box term by parts:
#   e^{(D-2)sigma} box(sigma) -> -(D-2) e^{(D-2)sigma} (d sigma)^2 (+ total deriv.)
coeff_box  = -2*(D-1)
coeff_grad = -(D-1)*(D-2)
coeff_after_ibp = sp.expand(coeff_grad + (-coeff_box)*(D-2)/1)  # -(-2(D-1))*(D-2)
assert sp.simplify(coeff_after_ibp - (D-1)*(D-2)) == 0
ok("total (d sigma)^2 coefficient after integration by parts: (D-1)(D-2)")

assert (D-1)*(D-2) == 6 if False else sp.simplify(((D-1)*(D-2)).subs(D, 4)) == 6
Mpl, phi = sp.symbols('M_Pl varphi', positive=True)
kinetic = sp.Rational(1, 2)*Mpl**2*6*sigma**2       # schematic: (d sigma)^2 -> sigma^2
canonical = sp.Rational(1, 2)*phi**2
assert sp.simplify(kinetic.subs(sigma, phi/(sp.sqrt(6)*Mpl)) - canonical) == 0
ok("coefficient 6 at D=4 and varphi = sqrt(6) M_Pl sigma canonical")

# ----------------------------------------------------------------------
# [5] Equilibrium correction c_d^eq (Eq. 12) and decomposition (Remark 4)
# ----------------------------------------------------------------------
# Averaging over a geodesic ball: <x^i x^j> = delta^ij l^2/(d+2), so with
# s = l^2 nabla^2 sigma / (2(d+2)) the ball averages are
#   <e^{a sigma(x)}> = e^{a sigma_0} (1 + a s) + O(l^4).
# Stationarity of <F> in sigma_0 then shifts the equilibrium point.
d_, s = sp.symbols('d s', positive=True)
c1, c2 = sp.symbols('c_1 c_2', positive=True)
s0 = sp.symbols('sigma_0')

Favg = c1*sp.exp(d_*s0)*(1 + d_*s) + c2*sp.exp(-s0)*(1 - s)
st0 = sp.solve(sp.diff(Favg, s0), s0)[0]
ratio = sp.exp((d_+1)*st0)                       # equals I * (1 + c_d^eq l^2 nabla^2 sigma)
I0 = c2/(c1*d_)                                  # uncorrected equilibrium
corr = sp.series(sp.simplify(ratio/I0), s, 0, 2).removeO()
# corr = 1 + (coefficient) * s ; with s = l^2 nabla^2 sigma / (2(d+2)):
coef_s = sp.simplify(sp.diff(corr, s).subs(s, 0))
c_d_eq = sp.simplify(coef_s/(2*(d_+2)))          # coefficient of l^2 nabla^2 sigma
assert sp.simplify(c_d_eq - (-(d_+1)/(2*(d_+2)))) == 0
ok("c_d^eq = -(d+1)/(2(d+2)), = -2/5 at d=3")
assert sp.simplify(c_d_eq.subs(d_, 3) + sp.Rational(2, 5)) == 0

c_d_full = -(3*d_**2 + d_ + 2)/(6*d_*(d_+2))
dressing = (d_-1)/(3*d_*(d_+2))
assert sp.simplify(c_d_full - (-(d_+1)/(2*(d_+2)) + dressing)) == 0
assert sp.simplify(c_d_full.subs(d_, 3) + sp.Rational(16, 45)) == 0
assert dressing.subs(d_, 1) == 0
ok("c_d = c_d^eq + (d-1)/(3d(d+2)); c_3 = -16/45; dressing vanishes at d=1")

print("\nAll equilibrium checks passed.")
