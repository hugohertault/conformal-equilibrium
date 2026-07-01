"""Direct verification of the conformal transform of the Ricci scalar (Lemma 2):

    g_{mu nu} = e^{2 sigma} eta_{mu nu}   (flat reference, D = 4)
    R = e^{-2 sigma} [ -2(D-1) box(sigma) - (D-1)(D-2) (d sigma)^2 ]

with box and (d sigma)^2 computed with the flat metric eta. The Ricci scalar is
computed from scratch (Christoffels -> Riemann -> Ricci -> R) for a generic
sigma(x^0..x^3); the difference with the closed form is simplified to zero.
"""
import sympy as sp

D = 4
x = sp.symbols('x0:4', real=True)
sigma = sp.Function('sigma')(*x)

eta = sp.diag(-1, 1, 1, 1)
g = sp.exp(2*sigma)*eta
ginv = g.inv()

def christoffel(g, ginv, coords):
    n = len(coords)
    Gamma = [[[sp.S(0)]*n for _ in range(n)] for _ in range(n)]
    for l in range(n):
        for m in range(n):
            for n_ in range(n):
                expr = sp.S(0)
                for r in range(n):
                    expr += ginv[l, r]*(sp.diff(g[r, m], coords[n_])
                                        + sp.diff(g[r, n_], coords[m])
                                        - sp.diff(g[m, n_], coords[r]))
                Gamma[l][m][n_] = sp.simplify(expr/2)
    return Gamma

print("computing Christoffels ...")
Gamma = christoffel(g, ginv, x)

print("computing Ricci tensor ...")
Ric = sp.zeros(D, D)
for m in range(D):
    for n_ in range(D):
        expr = sp.S(0)
        for l in range(D):
            expr += sp.diff(Gamma[l][m][n_], x[l]) - sp.diff(Gamma[l][m][l], x[n_])
            for r in range(D):
                expr += Gamma[l][l][r]*Gamma[r][m][n_] - Gamma[l][n_][r]*Gamma[r][m][l]
        Ric[m, n_] = expr

R_scalar = sp.S(0)
for m in range(D):
    for n_ in range(D):
        R_scalar += ginv[m, n_]*Ric[m, n_]
R_scalar = sp.simplify(R_scalar)

# closed form of Lemma 2 with hat-R = 0 (flat reference)
etainv = eta.inv()
box_sigma = sum(etainv[a, a]*sp.diff(sigma, x[a], 2) for a in range(D))
grad2 = sum(etainv[a, a]*sp.diff(sigma, x[a])**2 for a in range(D))
R_formula = sp.exp(-2*sigma)*(-2*(D-1)*box_sigma - (D-1)*(D-2)*grad2)

diff = sp.simplify(R_scalar - R_formula)
assert diff == 0, f"nonzero difference: {diff}"
print("  [PASS] R(computed) - R(Lemma 2) = 0 identically, generic sigma(x), D=4")
print("\nRicci conformal-transform check passed.")
