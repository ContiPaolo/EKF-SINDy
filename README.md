Source code of the papers:
- [1] **"EKF–SINDy: Empowering the extended Kalman filter with sparse identification of nonlinear dynamics"** open access on [CMAME](https://www.sciencedirect.com/science/article/pii/S0045782524005206) journal.
- [2] **"Online learning in bifurcating dynamic systems via SINDy and Kalman filtering"** preprint available on [arXiv](https://arxiv.org/abs/2411.04842)

![graphical_abstract-ezgif com-crop](https://github.com/ContiPaolo/EKF-SINDy/assets/51111500/d94bc746-9b4f-4830-a5b3-8ed06041652f)

## Test cases 
#### Online model estimation:
- Selkov model ([2], Sect. 3.2):
  - Modeling and state estimation of systems undergoing Hopf bifurcations `CSelkovModel_OnlineLearning.ipynb`
    
#### Parameter estimation:
- Partially observed nonlinear system ([1], Sect. 3.2):
  - Estimation of stifness of hidden oscillator (Sect. 3.2.3) `CoupledOscillators_StifnessEstimation.ipynb`
  - Estimation of linear and quadratic coupling coefficients (Sect. 3.2.4.) `CoupledOscillators_CouplingEstimation.ipynb`
  
- Shear building under seismic excitations ([1], Sect. 3.1)
  - Estimation of the inter-storey stiffness `ShearBuilding_test\ShearBuilding.ipynb`
    - Train/test data are available [here](https://zenodo.org/records/11581079).

## Abstract
We propose a framework that combines the Extended Kalman Filter (EKF) with Sparse Identification of Nonlinear Dynamics (SINDy) for online data assimilation and model adaptation of nonlinear dynamic systems. SINDy is employed to identify the system dynamics directly from data, thus mitigating potential biases due to incorrect modeling assumptions and providing a computationally efficient, physically interpretable model. By treating states, parameters and SINDy model coefficients as random variables, the EKF enables joint state-parameter estimation, allowing real-time updates to adapt the model to time-varying conditions or unforeseen events. This approach not only facilitates efficient model sensitivity evaluation but also ensures robust operation beyond the original training range of SINDy. Overall, this method offers a data-driven, easy-to-apply strategy for dynamic model identification, making it well-suited for handling time-varying, nonlinear systems with reduced epistemic uncertainty.
