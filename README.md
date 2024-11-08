> :warning: The code for the paper **"Online learning in bifurcating dynamic systems via SINDy and Kalman filtering"** will be available soon.

Source code of the papers **"EKF–SINDy: Empowering the extended Kalman filter with sparse identification of nonlinear dynamics"** 
![graphical_abstract-ezgif com-crop](https://github.com/ContiPaolo/EKF-SINDy/assets/51111500/d94bc746-9b4f-4830-a5b3-8ed06041652f)



## Reference
The pulbished version of "EKF–SINDy: Empowering the extended Kalman filter with sparse identification of nonlinear dynamics" is available on [CMAME](https://www.sciencedirect.com/science/article/pii/S0045782524005206).
The preprint of "Online learning in bifurcating dynamic systems via SINDy and Kalman filtering" is available on [arXiv](https://arxiv.org/abs/2411.04842).

## Test cases
- Partially observed nonlinear system (Sect. 3.2):
  - Estimation of stifness of hidden oscillator (Sect. 3.2.3) `CoupledOscillators_StifnessEstimation.ipynb`
  - Estimation of linear and quadratic coupling coefficients (Sect. 3.2.4.) `CoupledOscillators_CouplingEstimation.ipynb`
  
- Shear building under seismic excitations (Sect. 3.1)
  - Estimation of the inter-storey stiffness `ShearBuilding_test\ShearBuilding.ipynb`
    - Train/test data are available [here](https://zenodo.org/records/11581079).


## Abstract
Observed data from a dynamic system can be assimilated into a predictive model by means of Kalman filters. Nonlinear extensions of the Kalman filter, such as the Extended Kalman Filter (EKF), are required to enable the joint estimation of (possibly nonlinear) system dynamics and of input parameters. To construct the evolution model used in the prediction phase of the EKF, we propose to rely on the Sparse Identification of Nonlinear Dynamics (SINDy). The numerical integration of a SINDy model leads to great computational savings compared to alternate strategies based on, e.g., finite elements. Indeed, SINDy allows for the immediate definition of the Jacobian matrices required by the EKF to identify system dynamics and properties, a derivation that is usually extremely involved with physical models. As a result, combining the EKF with SINDy provides a computationally efficient, easy-to-apply approach for the identification of nonlinear systems, capable of robust operation even outside the range of training of SINDy. To demonstrate the potential of the approach, we address the identification of a linear non-autonomous system consisting of a shear building model excited by real seismograms, and the identification of a partially observed nonlinear system. The challenge arising from applying SINDy when the system state is not accessible has been relieved by means of time-delay embedding. The great accuracy and the small uncertainty associated with the state identification, where the state has been augmented to include system properties, underscores the great potential of the proposed strategy, paving the way for the development of predictive digital twins in different fields.


