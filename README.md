# DroTrack

**DroTrack: High-speed Drone-based Object Tracking Under Uncertainty**<br />
[Ali Hamdi](https://scholar.google.com.au/citations?user=Q5qW1rcAAAAJ&hl=en),[Flora Salim](http://florasalim.com/), [Du Yong Kim](https://sites.google.com/site/duyongkim/)<br />
**FUZZ-IEEE 2020** <br />
**[[Paper](https://arxiv.org/abs/2005.00828)] [[Video](https://youtu.be/i0oiodX9o6g)]** <br />
**Finalist of best student paper award in IEEE-FUZZ 2020.<br />

DroTrack is a high-speed visual single-object tracking framework for drone-captured video sequences. The complex motion of drones, i.e., multiple degrees of freedom in three-dimensional space, causes high uncertainty. The uncertainty problem leads to inaccurate location predictions and fuzziness in scale estimations. DroTrack solves such issues by discovering the dependency between object representation and motion geometry. DroTrack has been evaluated using two datasets of 51,462 drone-captured frames. The combination of the FCM segmentation and the angular scaling increased DroTrack precision by up to 9% and decreased the centre location error by 162 pixels on average. DroTrack outperforms all the high-speed trackers and achieves comparable results in comparison to deep learning trackers. DroTrack offers high frame rates up to 1000 frame per second (fps) with the best location precision, more than a set of state-of-the-art real-time trackers.

### Bibtex
If you find this code useful, please consider citing:

```
@inproceedings{hamdi2020drotrack,
    author    = {Hamdi, Ali and Salim, Flora and Kim,Du Yong},
    title     = {DroTrack: High-speed Drone-based Object Tracking Under Uncertainty},
    booktitle = {29th {IEEE} International Conference on Fuzzy Systems, {FUZZ-IEEE} 2020, Glasgow, UK, July 19-24, 2020},
    pages     = {1--8},
    publisher = {{IEEE}},
    year      = {2020},
    url       = {https://doi.org/10.1109/FUZZ48607.2020.9177571},
    doi       = {10.1109/FUZZ48607.2020.9177571}
}
```
<br />

<div align="center">
  <img src="code/src/DroTrack.png" width="600px" />
</div>

### Results

<div align="center">
  <img src="code/src/Final-results.jpg" width="600px" />
</div>

<div align="center">
  <img src="code/src/Final-frames.jpg" width="600px" />
</div>

### Demo
<div align="center">
    <img src="demo/DroTrack_demo_1.gif" width="600" />
    <img src="demo/DroTrack_demo_2.gif" width="600" />
    <img src="demo/DroTrack_demo_3.gif" width="600" />
    <img src="demo/DroTrack_demo_4.gif" width="600" />
    <img src="demo/DroTrack_demo_5.gif" width="600" />
    <img src="demo/DroTrack_demo_6.gif" width="600" />
    <img src="demo/DroTrack_demo_7.gif" width="600" />
    <img src="demo/DroTrack_demo_8.gif" width="600" />
    <img src="demo/DroTrack_demo_9.gif" width="600" />
    <img src="demo/DroTrack_demo_10.gif" width="600" />
    <img src="demo/DroTrack_demo_11.gif" width="600" />
    <img src="demo/DroTrack_demo_12.gif" width="600" />
    <img src="demo/DroTrack_demo_13.gif" width="600" />
    <img src="demo/DroTrack_demo_14.gif" width="600" />
</div>

**[[Video](https://youtu.be/i0oiodX9o6g)]**

## License
Licensed under an MIT license.
