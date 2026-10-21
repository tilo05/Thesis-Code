# Sparse Grid Methods for PDE-Based Pricing of Bermudan Swaptions in the LIBOR Market Model

European and Bermudan swaptions are priced in the LIBOR market model by solving the PCA-reduced pricing PDE
with the Douglas ADI scheme on full grids (1-2 components) and on sparse grids (3-5 components) with the sparse grid
combination technique.
Monte Carlo reference prices are computed with MLMC (European) and LSMC (Bermudan).

## Structure

- `common/`: setup of the LMM (`lmm.py`), yield curve (`YieldCurve.py`), drift and payoffs (`lmm_functions.py`)
  and the tridiagonal solver (`solver.py`).
- `PDE/`: full grid PDE solver in PCA coordinates (`PDESolverFullGrid_PCA.py`), combination technique
  (`combination_technique.py`) and full/sparse grid pricing drivers (`PCA_driver.py`).
- `MonteCarlo/`: path simulation with Euler/Milstein (`paths.py`), MLMC for European (`MLMC.py`) and
  LSMC for Bermudan swaptions (`LSMC.py`).
- `test/`: `generate_thesis_results.py` generates all numerical results of the thesis.

## Installation

Developed with Python 3.14.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

The `__main__` block of `generate_thesis_results.py` contains one block per experiment of the thesis
(convergence orders, MLMC/LSMC reference prices, PDE prices of European and Bermudan swaptions,
sensitivity-based truncation check, level analysis, exercise boundary). Comment out blocks in the 
`__main__` block to run single experiments. The results are written as `.dat` files to `data/`.
Note that the Monte Carlo simulations are not seeded by default, so repeated runs vary within the reported 95% confidence intervals.