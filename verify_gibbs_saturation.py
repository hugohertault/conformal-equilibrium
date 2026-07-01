"""Verification of the Gibbs / saturation results (Sec. 6) and the horizon
temperature algebra (Secs. 7-8).

Checks:
  [1] The modular-energy minimiser at fixed entropy is the Gibbs family (Prop. 4)
  [2] Exact identity:  Delta<K> - Delta S = S_rel(rho_beta || rho_0)      (Thm. 2)
      verified numerically on an explicit modular spectrum, to ~1e-12
  [3] Leading deficit: S_rel -> (Delta S)^2 / (2C), C = Var_vac(K),
      ratio -> 1 as the excitation -> 0                                   (Thm. 2)
  [4] S_rel > 0 strictly for beta != 1 (Klein): the relative-entropy step
      of the no-conical-defect theorem                                    (Thm. 3)
  [5] T_H = hbar kappa / (2 pi k_B c) -> hbar c^3/(8 pi G M k_B)          (Sec. 7)
"""
import numpy as np
import sympy as sp

ok = lambda name: print(f"  [PASS] {name}")

# ----------------------------------------------------------------------
# [1] Gibbs minimiser (Proposition 4) - symbolic stationarity
# ----------------------------------------------------------------------
p, k, mu, beta = sp.symbols('p k mu beta', positive=True)
# Lagrangian density per eigenvalue of the constrained minimisation
#   min <K>  s.t.  Tr rho = 1,  S(rho) = S*   =>   L = p k - mu p - beta S
# with S = -p ln p, i.e. L = p*k - mu*p + beta*p*log(p)
L = p*k - mu*p + beta*p*sp.log(p)
stat = sp.solve(sp.diff(L, p), p)
assert len(stat) == 1
p_star = stat[0]
# p* proportional to exp(-k/beta): the ratio p*/exp(-k/beta) is k-independent
ratio = sp.simplify(p_star/sp.exp(-k/beta))
assert sp.diff(ratio, k) == 0
ok("stationarity gives p_i proportional to exp(-k_i/beta): the Gibbs family")

# ----------------------------------------------------------------------
# [2]-[4] Numerics on an explicit modular spectrum
# ----------------------------------------------------------------------
rng = np.random.default_rng(20260701)
K = np.sort(rng.exponential(scale=1.0, size=400))      # generic positive spectrum

def gibbs(beta):
    w = np.exp(-beta*K)
    return w/w.sum()

def entropy(rho):
    r = rho[rho > 0]
    return -(r*np.log(r)).sum()

rho0 = gibbs(1.0)                                       # 'vacuum' (beta = 1)
S0, K0 = entropy(rho0), (rho0*K).sum()
C = (rho0*K**2).sum() - K0**2                           # Var_vac(K)

max_err, ratios = 0.0, []
for beta_v in [1.5, 1.2, 1.05, 1.01, 1.001, 0.999, 0.99, 0.95, 0.8]:
    rb = gibbs(beta_v)
    dK = (rb*K).sum() - K0
    dS = entropy(rb) - S0
    deficit = dK - dS
    srel = (rb*(np.log(rb) - np.log(rho0))).sum()       # S_rel(rho_beta || rho_0)
    max_err = max(max_err, abs(deficit - srel))
    if abs(dS) > 0:
        ratios.append((abs(beta_v - 1), srel/(dS**2/(2*C))))

assert max_err < 1e-12, max_err
ok(f"Delta<K> - Delta S = S_rel exactly (max deviation {max_err:.2e})")

ratios.sort()
r_small = ratios[0][1]      # smallest excitation
r_large = ratios[-1][1]
assert abs(r_small - 1) < 1e-3, r_small
assert abs(r_small - 1) < abs(r_large - 1)
ok(f"S_rel / [(Delta S)^2/2C] -> 1 as excitation -> 0 "
   f"(={r_small:.6f} at |beta-1|={ratios[0][0]})")

for beta_v in [0.5, 0.9, 1.1, 2.0]:
    rb = gibbs(beta_v)
    srel = (rb*(np.log(rb) - np.log(rho0))).sum()
    assert srel > 0
ok("S_rel(rho_beta || rho_0) > 0 strictly for beta != 1 (Klein's inequality)")

# ----------------------------------------------------------------------
# [5] Hawking / Unruh temperature algebra
# ----------------------------------------------------------------------
hbar, c, Gn, M, kB = sp.symbols('hbar c G M k_B', positive=True)
kappa = c**4/(4*Gn*M)                                   # Schwarzschild surface gravity
T_H = hbar*kappa/(2*sp.pi*kB*c)
assert sp.simplify(T_H - hbar*c**3/(8*sp.pi*Gn*M*kB)) == 0
ok("T_H = hbar kappa / (2 pi k_B c) = hbar c^3 / (8 pi G M k_B)")

print("\nAll Gibbs/saturation/temperature checks passed.")
