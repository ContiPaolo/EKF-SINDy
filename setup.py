[metadata]
name = EKFSINDy - Paolo Conti
version = 1.0
author = Paolo Conti
author_email = paolo.conti@polimi.it
description = A package for empowering extended Kalman filter (EKF) with Sparse Identification of Nonlinear Dynamics (SINDy)
long_description = file: README.md
long_description_content_type = text/markdown
url = https://github.com/NicolaRFranco/dlrom/
project_urls =
    Bug Tracker = https://github.com/ContiPaolo/EKF-SINDy/
classifiers =
    Programming Language :: Python :: 3
    License :: OSI Approved :: MIT License
    Operating System :: Linux

[options]
package_dir =
    = src
packages = find:
python_requires = >=3.6
install_requires =
    pysindy
    numpy
    matplotlib
    scipy
    sklearn
    pyDOE
