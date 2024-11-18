Source code of the papers:
[[1]].  **"EKF–SINDy: Empowering the extended Kalman filter with sparse identification of nonlinear dynamics"** open access on [CMAME](https://www.sciencedirect.com/science/article/pii/S0045782524005206) journal.
[[2]]. **"Online learning in bifurcating dynamic systems via SINDy and Kalman filtering"** preprint available on [arXiv](https://arxiv.org/abs/2411.04842)

![graphical_abstract-ezgif com-crop](https://github.com/ContiPaolo/EKF-SINDy/assets/51111500/d94bc746-9b4f-4830-a5b3-8ed06041652f)

## Test cases 
#### Online model estimation:
- Selkov model (Paper 2, Sect. 3.2):
  - Modeling and state estimation of systems undergoing Hopf bifurcations `CSelkovModel_OnlineLearning.ipynb`
    
#### Parameter estimation:
- Partially observed nonlinear system (Paper 1, Sect. 3.2):
  - Estimation of stifness of hidden oscillator (Sect. 3.2.3) `CoupledOscillators_StifnessEstimation.ipynb`
  - Estimation of linear and quadratic coupling coefficients (Sect. 3.2.4.) `CoupledOscillators_CouplingEstimation.ipynb`
  
- Shear building under seismic excitations (Paper 1, Sect. 3.1)
  - Estimation of the inter-storey stiffness `ShearBuilding_test\ShearBuilding.ipynb`
    - Train/test data are available [here](https://zenodo.org/records/11581079).

