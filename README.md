# conformal-equilibrium

Verification scripts for the paper **"The conformal mode as an entropic
equilibrium: a variational origin of the relation e^{Dσ} = S_ent/S_Bek"**
(H. Hertault, submitted to *General Relativity and Gravitation*).

Every symbolic and numerical identity claimed in the paper is reproduced
here independently with SymPy and NumPy.

## Contents

| Script | Verifies |
|---|---|
| `verify_equilibrium.py` | Stationary point and convexity of the free energy (Prop. 1); the Bekenstein normalisation ⇔ holographic vacuum condition, both directions (Prop. 2); exponent rigidity (Sec. 3.1); kinetic coefficient (D−1)(D−2) and the canonical normalisation φ = √6 M_Pl σ (Sec. 5); the equilibrium correction c_d^eq = −(d+1)/(2(d+2)) and the decomposition of the consolidated coefficient c_d (Eq. 12, Remark 4). |
| `verify_ricci.py` | The conformal transform of the Ricci scalar (Lemma 2) by direct component computation (Christoffels → Riemann → Ricci → R) for a generic σ(x) on a flat reference in D = 4; the difference with the closed form simplifies to zero identically. |
| `verify_gibbs_saturation.py` | The modular-energy minimiser at fixed entropy is the Gibbs family (Prop. 4); the exact identity Δ⟨K⟩ − ΔS = S_rel(ρ_β‖ρ_0) on an explicit modular spectrum, to ~10⁻¹²; the quadratic deficit (ΔS)²/2C with ratio → 1 as the excitation → 0 (Thm. 2); strict positivity of S_rel for β ≠ 1, the relative-entropy step of the no-conical-defect theorem (Thm. 3); the Hawking-temperature algebra (Sec. 7). |

## Running

```bash
pip install -r requirements.txt
python verify_equilibrium.py
python verify_ricci.py          # a few minutes: full component computation
python verify_gibbs_saturation.py
```

Each script prints `[PASS]` per check and exits with an assertion error on
any failure.

## Requirements

- Python ≥ 3.10
- sympy ≥ 1.12
- numpy ≥ 1.24

## License

MIT — see [LICENSE](LICENSE).
