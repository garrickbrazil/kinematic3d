
# Setup for Kinematic3D

This document acts as a (suggested) guide for setting up cuda, Python 3, and Anaconda. If components are already setup please feel encouraged to skip sections or use any alternative methods such as pip. 

*Note:* there are MANY alternative methods to install all below packages. This guide is only an example!

#### Install cuda

Follow the official instructions to install cuda 10 by visiting [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

#### Install Anaconda / Python 3.6.5

For Kinematic3D we utilized Python 3.6.5 Anaconda. Other versions may also work.

1. Install your preferred version of Anaconda
	```
	cd ~
	wget https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh
	sh Anaconda3-5.2.0-Linux-x86_64.sh
	```

	Defaults are usually fine. 
	Recommend letting the installer add to your path and avoid microsoft extention (unless on windows). 
	Before doing any of the below make sure that the path is setup properly:
	
    ```
    python --version
    ```
	Hopefully you see Python 3.6.5, Anaconda Inc.
    
    You may also want to make a new environment and then source it by
    ```
    conda create -n kinematic3d python=3.6.5
    source activate kinematic3d
    ```

1. Install general python packages
	
	```
	conda install -c menpo opencv3=3.1.0 openblas
	conda install cython scikit-image
    conda install -c conda-forge easydict shapely
	```
    
1. Install pytorch

	```
    conda install pytorch=1.2.0 torchvision=0.4.0 cudatoolkit=10.2 -c pytorch
	```



