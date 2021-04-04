Official implementation of  **[Cross-Domain Face Synthesis using a Controllable GAN](http://openaccess.thecvf.com/content_WACV_2020/html/Mokhayeri_Cross-Domain_Face_Synthesis_using_a_Controllable_GAN_WACV_2020_paper.html)**
===========

This page contains end-to-end demo code that generates a set of synthetic face images under a specified pose from an unconstrained 2D face image based on the information obtained from target domain.  

## Prerequisite

Download the [Basel Face Model](https://faces.dmi.unibas.ch/bfm/)** and move `01_MorphableModel.mat` into the folder.

## Instructions
__Install:__

```
conda create -n CGAN python=3.6
conda activate CGAN

pip install \
pyglet \
pywavefront \
opencv-python \
imutils  \
 https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f


pip install \
    matplotlib \
    keras==2.2.5 \
    tensorflow-gpu==1.14

pip install git+https://www.github.com/keras-team/keras-contrib.git
```

- **Generate 3D simulated images from still images:**

Put the still images in "./face3d/input/", while each identity is in a seperate folder.
Run:
```
cd face3d
pyhton face3d.py
cd ..
```
3D rendered results will be in:

```
"face3d/output"
```

- **Use C-GAN to refine the 3D simulated images:**

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


@InProceedings{Mokhayeri_2020_WACV,
author = {Mokhayeri, Fania and Kamali, Kaveh and Granger, Eric},
title = {Cross-Domain Face Synthesis using a Controllable GAN},
booktitle = {The IEEE Winter Conference on Applications of Computer Vision (WACV)},
year = {2020}
}
