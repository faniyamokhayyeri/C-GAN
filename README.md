Official implementation of  **[Cross-Domain Face Synthesis using a Controllable GAN](http://openaccess.thecvf.com/content_WACV_2020/html/Mokhayeri_Cross-Domain_Face_Synthesis_using_a_Controllable_GAN_WACV_2020_paper.html)**
===========

This page contains end-to-end demo code that generates a set of synthetic face images under a specified pose from an unconstrained 2D face image based on the information obtained from target domain.  

## Prerequisite

Download the [Basel Face Model](https://faces.dmi.unibas.ch/bfm/) and move `01_MorphableModel.mat` into the folder.

## Instructions:
__Install:__
1. keras
2. tensorflow-gpu
3. scipy 1.1.0
4. scikit-image

```
sudo apt-add-repository ppa:zarquon42/meshlab
sudo apt-get update
sudo apt-get install meshlab=1.3.2+dfsg1-2build4
```

```
pip install git+https://www.github.com/keras-team/keras-contrib.git
```

__ Generate 3D simulated images from still images:__

Put the still images in "./face3d/input/", while each identity is in a seperate folder.
Run:
```
pyhton face3d.py
```
3D rendered results will be in:

```
"face3d/output"
```

__Use CGAN to refine the 3D simulated images:__

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

Results will be in:
```
"./output"
```

## Data

- **[ChokePoint](http://arma.sourceforge.net/chokepoint/)** 
- **[COX-S2V](http://vipl.ict.ac.cn/view_database.php?id=3)** 

## Citation

If you find this work useful, please cite our paper with the following bibtex:

@article{mokhayeri2018domain,
  title={Cross-Domain Face Synthesis using a Controllable GAN},
  author={Mokhayeri, Fania and Granger, Eric and Kamali, Kaveh},
  booktitle={WACV},
  year={2020},
  publisher={IEEE}
}

