# Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation (CVPR 2021)

[Hang Zhou](https://hangz-nju-cuhk.github.io/), Yasheng Sun, [Wayne Wu](https://wywu.github.io/), [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/), [Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/), and [Ziwei Liu](https://liuziwei7.github.io/).

[[Project]](https://hangz-nju-cuhk.github.io/projects/PC-AVS) [[Paper]](https://arxiv.org/abs/1807.07860) [[Demo]](https://www.youtube.com/watch?v=-J2zANwdjcQ)

[comment]: <> (<img src='./misc/teaser.png' width=880>)


<img src='./misc/demo.gif' width=720>

We propose **Pose-Controllable Audio-Visual System (PC-AVS)**, 
which achieves free pose control when driving arbitrary talking faces with audios. Instead of learning pose motions from audios, we leverage another pose source video to compensate only for head motions.


<img src='./misc/method.png' width=880>

## Requirements
* Python 3.6 and [Pytorch](https://pytorch.org/) 1.3.0 are used. Basic requirements are listed in the 'requirements.txt'.

```
pip install -r requirements.txt
```


## Quick Start: Generate Demo Results
* Download the pre-trained models from the [checkpoints](https://drive.google.com/file/d/1Zehr3JLIpzdg2S5zZrhIbpYPKF-4gKU_/view?usp=sharing).

* Create the default folder ```./checkpoints``` and 
unzip ```demo.zip``` within ```checkpoints```. There should be a ```demo``` folder containing 5 ```pth```s.

* Download the examples, and unzip them in the ```misc``` folder.

* Run the demo scripts:
``` bash
bash experiments/demo_vox.sh
```


* The ```--gen_video``` argument is by default on, 
```ffmpeg``` is required to use this flag in linux systems. 
A ```avconcat.mp4``` will be saved in the ```./results``` folder 
in the form of following:

<img src='./misc/output.gif' width=320>

From left to right are the input reference, the generated results,
the pose source video and the original video synced with the driving audio.

## Preparing Testing Data
* Coming soon

## Train Your Own Model
* Coming soon

## License and Citation
The use of this software is RESTRICTED to **non-commercial research and educational purposes**.

```
@InProceedings{Zhou2021pose,
author = {Zhou, Hang and Sun, Yasheng and Wu, Wayne and Loy, Chen Change and Wang, Xiaogang and Liu, Ziwei},
title = {Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2021}
}
```

## Acknowledgement
* The structure of this codebase is borrowed from [SPADE](https://github.com/NVlabs/SPADE).
