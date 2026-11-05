Source code of the papers:
- [1] **"EKF–SINDy: Empowering the extended Kalman filter with sparse identification of nonlinear dynamics"** open access on [CMAME](https://www.sciencedirect.com/science/article/pii/S0045782524005206)
![graphical_abstract-ezgif com-crop](https://github.com/ContiPaolo/EKF-SINDy/assets/51111500/d94bc746-9b4f-4830-a5b3-8ed06041652f)
- [2] **"Online learning in bifurcating dynamic systems via SINDy and Kalman filtering"** open access on [Nonlinear Dynamics](https://link.springer.com/article/10.1007/s11071-025-11029-y)
![Online_EKFSINDy_crop](https://github.com/user-attachments/assets/e163aa3c-26f6-40bb-8ccd-c4a4de721210)

## Test cases
#### Online model estimation:
- Selkov model ([2], Sect. 3.2):
  - Modeling and state estimation of systems undergoing Hopf bifurcations `notebooks/SelkovModel_OnlineLearning.ipynb`

#### Parameter estimation:
- Partially observed nonlinear system ([1], Sect. 3.2):
  - Estimation of stifness of hidden oscillator (Sect. 3.2.3) `notebooks/CoupledOscillators_StifnessEstimation.ipynb`
  - Estimation of linear and quadratic coupling coefficients (Sect. 3.2.4.) `notebooks/CoupledOscillators_CouplingEstimation.ipynb`

- Shear building under seismic excitations ([1], Sect. 3.1)
  - Estimation of the inter-storey stiffness `notebooks/ShearBuilding.ipynb`
    - Train/test data are available [here](https://zenodo.org/records/11581079): put the csv files in `data/` (or set `EKFSINDY_DATA` to their folder).

## Installation
The project is a Python package (`ekfsindy`, Python >= 3.10, pysindy 2, NumPy 2) managed with
[uv](https://docs.astral.sh/uv/); `uv.lock` pins the exact versions used for the results:

```bash
uv sync                       # creates .venv with the package, Jupyter and dev dependencies
uv run jupyter lab            # run the notebooks
uv run pytest                 # unit and regression tests
```

On Google Colab, the first cell of each notebook installs the package with
`pip install "ekfsindy @ git+https://github.com/ContiPaolo/EKF-SINDy"`; no restart is needed.

## Package overview
```text
src/ekfsindy/
  library.py      SymbolicLibrary: SINDy features as SymPy expressions, compiled to JAX
  model.py        SparseModel: dx/dt = Xi @ Theta(x, p, u), with online-estimated coefficients
  ekf.py          AugmentedEKF: EKF on z = [x, p, xi], Jacobians by automatic differentiation
  observation.py  observation maps h(z, u)
  data.py         noise, time-delay embedding, shear-building data loader
  systems.py      ground-truth ODEs of the examples
  plotting.py     estimate / credibility-band plots
tests/            unit tests, regression tests against the original implementation (tests/legacy)
```

The library is symbolic, so derivatives never have to be written by hand: `jax.jacfwd` differentiates
the model with respect to states, parameters and SINDy coefficients alike.

```python
import numpy as np
import pysindy as ps
from ekfsindy import AugmentedEKF, SparseModel, SymbolicLibrary, observation

sindy = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3))
sindy.fit(X_train, t=dt, feature_names=["u1", "u2"])

library = SymbolicLibrary.from_feature_names(sindy.get_feature_names(), ["u1", "u2"])
library.symbolic_jacobian()                      # SymPy matrix d(theta)/d(u1, u2)

# track every nonzero SINDy coefficient online
model = SparseModel.from_pysindy(sindy, library, estimated="nonzero")
ekf = AugmentedEKF(model, observation.select_states([0, 1]), Q=Q, R=R, dt=dt, integrator="rk4")
result = ekf.run(model.augmented_state(Y[0]), P0, Y)
lower, upper = result.bounds(1.96)
```

## Abstract
We propose a framework that combines the Extended Kalman Filter (EKF) with Sparse Identification of Nonlinear Dynamics (SINDy) for online data assimilation and model adaptation of nonlinear dynamic systems. SINDy is employed to identify the system dynamics directly from data, thus mitigating potential biases due to incorrect modeling assumptions and providing a computationally efficient, physically interpretable model. By treating states, parameters and SINDy model coefficients as random variables, the EKF enables joint state-parameter estimation, allowing real-time updates to adapt the model to time-varying conditions or unforeseen events. This approach not only facilitates efficient model sensitivity evaluation but also ensures robust operation beyond the original training range of SINDy. Overall, this method offers a data-driven, easy-to-apply strategy for dynamic model identification, making it well-suited for handling time-varying, nonlinear systems with reduced epistemic uncertainty.
