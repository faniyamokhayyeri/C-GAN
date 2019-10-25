# CGAN
Make realistic face images from 3D simulated


## Paper:

Official implementation of "Cross-Domain Face Synthesis using a Controllable GAN" WACV 2020.


## Instructions:
__Install:__
* keras
* tensorflow-gpu
* scipy 1.1.0
* scikit-image

```
sudo apt-add-repository ppa:zarquon42/meshlab
sudo apt-get update
sudo apt-get install meshlab=1.3.2+dfsg1-2build4
```


```
pip install git+https://www.github.com/keras-team/keras-contrib.git
```

__Generating 3D simulated images from still images:__

* Put the still images in "./face3d/input/", while each identity is in a seperate folder.
* Run:

```
pyhton face3d.py
```

* 3D rendered results will be in "face3d/output"


__Using CGAN to refine the 3D simulated images:__

Put the train and test 3D simulated data in:

```
./data/chokepoint/train_sim
./data/chokepoint/train_sim
```

Put the train and test target data in:

```
data/chokepoint/train_target
data/chokepoint/train_target
```

Put the test and train labels (only 3D simulated data needs label) in:

```
./data/chokepoint/test_sim_labels.txt
./data/chokepoint/train_sim_labels.txt
```

Run:

```
python cgan.py
```

Results will be in "./output"
