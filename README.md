# Kinematic 3D Object Detection in Monocular Video


Garrick Brazil, Gerard Pons-Moll, Xiaoming Liu, Bernt Schiele

## Introduction


Source code as detailed in [arXiv report](https://arxiv.org/abs/2007.09548), in ECCV 2020. Please also visit our [project page](http://cvlab.cse.msu.edu/project-kinematic.html).

Much of the code in this project is a derivative of the code from [M3D-RPN](https://github.com/garrickbrazil/M3D-RPN), such that setup/organization is very similar. 

Our framework is implemented and tested with Ubuntu 16.04, Python 3, NVIDIA 1080 Ti GPU. Unless otherwise stated the below scripts and instructions assume working directory is the project root. 

If you utilize this framework, please cite our ECCV 2020 paper. 
	
    @inproceedings{brazil2020kinematic,
        title={Kinematic 3D Object Detection in Monocular Video},
        author={Brazil, Garrick and Pons-Moll, Gerard and Liu, Xiaoming and Schiele, Bernt},
        booktitle={Proceedings of European Conference on Computer Vision},
        address={Virtual},
        year={2020}
    }
    

## Setup

- **Cuda & Python**

    In this project we utilize Pytorch with Python 3, and a few Anaconda packages. Please review and follow this [installation guide](setup.md). However, feel free to try alternative versions or modes of installation. 

- **KITTI Data**

    Download the full [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) detection dataset including:
    
    - left color images of object data set (12 GB)
    - 3 temporally preceding frames (left color) (36 GB)
    - camera calibration matrices of object data set (16 MB)
    - training labels of object data set (5 MB)
    
    Then place a softlink (or the actual data) in `kinematic3d/data/kitti`. 

	```
    cd kinematic3d
	ln -s /path/to/kitti data/kitti
	```

	Then use the following scripts to extract the data splits, which use softlinks to the above directory for efficient storage. 

    ```
    python data/kitti_split1/setup_split.py
    ```
    
    Next, build the KITTI devkit eval for each split.

	```
	sh data/kitti_split1/devkit/cpp/build.sh
	```
- **KITTI Raw**

	Download the extra information such as extracted pose, calibration, and detection labels (when available) which we provide in [Kinematic3D-raw_extra.zip](https://www.cse.msu.edu/computervision/Kinematic3D-raw_extra.zip). Then extract this folder to `data/kitti/raw_extra` and `data/kitti_split1/raw_extra`. 

	We use Matlab and slightly modified versions of the KITTI matlab devkit to extract this information. If you wish to re-extract or otherwise, then you will need to download the calibration and tracklets from appropriate dates within the [raw dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php).
    
    Then you can modify and run the provided scripts in the matlab folder of the downloaded zip file:
    
    1. Specifically, the file `matlab/run_demoVehiclePath.m` will extract the poses ego-motion pose changes. 
    
    2. The file `matlab/convert_tracklets_to_detections.m` will extract the tracklet boxes (this is only needed IF you want to compute ground truth velocity, etc. These tracklets are not directly used otherwise).
    
- **Misc**

    Lastly, build the nms modules
    
    ```
	cd lib/nms
	make
	```

## Training

Training is split into a warmup, full, and pose. Review the configurations in `scripts/config` for details. 

``` 
# First train the warmup
python scripts/train_rpn_3d.py --config=kitti_3d_warmup

# Then train the model with uncertainty
python scripts/train_rpn_3d.py --config=kitti_3d_uncertainty

# Lastly train the full pose estimation 
python scripts/train_pose.py --config=kitti_3d_full
```

If your training is accidentally stopped, you can resume at a checkpoint based on the snapshot with the `restore` flag. 
For example to resume training starting at iteration 10k, use the following command.

```
python scripts/train_rpn_3d.py --config=kitti_3d_uncertainty --restore=10000
```

## Testing

We provide models for the main experiments on val1 / test data splits available to download here [Kinematic3D-Release.zip](https://www.cse.msu.edu/computervision/Kinematic3D-Release.zip).

Testing requires paths to the configuration file and model weights, exposed variables near the top `scripts/test_kalman.py`. To test a configuration and model, simply update the variables and run the test file as below. 

```
python scripts/test_kalman.py 
```

Similarly, we also provide a script to test only the 3D rpn as

```
python scripts/test_rpn_3d.py 
```

We also provide code to help with the visualization (for example in video). 

```
python scripts/tracking_video.py 
```

## Contact
For questions feel free to post here or directly contact the authors {[brazilga](http://garrickbrazil.com), [liuxm](http://www.cse.msu.edu/~liuxm/index2.html)}@msu.edu, {[gpons](https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html), [schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele)}@mpi-inf.mpg.de
