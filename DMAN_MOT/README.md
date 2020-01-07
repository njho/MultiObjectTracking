# Online Multi-Object Tracking with DMANs

This is the implementation of our ECCV 2018 paper [Online Multi-Object Tracking with Dual Matching Attention Networks](https://arxiv.org/abs/1902.00749). We integrate the ECO [1] for single object tracking. The code framework for MOT benefits from the MDP [2].

<p align="center">
  <img width="800" src="DMAN.png">
</p>
<p align="justify">

# Prerequisites
- Cuda 8.0
- Cudnn 5.1
- Python 2.7
- Keras 2.0.5
- Tensorflow 1.1.0

For example:
<pre><code>conda create -n mot anaconda python=2.7
conda activate mot
conda install -c menpo opencv
pip install tensorflow-gpu==1.1.0
pip install keras==2.0.5
</code></pre>

GoVertical Note:
This library also requires the following MatLab libraries to be installe: 
* Image Processing
* Image Acquisition
* Deep Learning
* Computer Vison
* Signal Processing
It also has a custom toolbox located here: [Piotr's Computer Vision Matlab Toolbox](https://pdollar.github.io/toolbox/)

When you download the MOT16 dataset, add a folder called and `MOT16` and place `train/` and `test/` inside for it to work


You also have to go to this link, download the zip: https://github.com/pdollar/toolbox.git
then `cp toolbox/classify/private/*` into `DMAN_MOT/ECO/external_libs/pdollar_toolbox/channels/private/*`


# Usage
1. Download the [DMAN model](https://zhiyanapp-build-release.oss-cn-shanghai.aliyuncs.com/zhuji_file/spatial_temporal_attention_model.h5) and put it into the "model/" folder.
2. Download the [MOT16 dataset](https://motchallenge.net/data/MOT16/), unzip it to the "data/" folder.
3. Cd to the "ECO/" folder, run the script install.m to compile libs for the ECO tracker
4. Run the socket server script:
<pre><code>python calculate_similarity.py
</code></pre>
5. Run the socket client script DMAN_demo.m in Matlab.
# Citation

If you use this code, please consider citing:

<pre><code>@inproceedings{zhu-eccv18-DMAN,
    author    = {Zhu, Ji and Yang, Hua and Liu, Nian and Kim, Minyoung and Zhang, Wenjun and Yang, Ming-Hsuan},
    title     = {Online Multi-Object Tracking with Dual Matching Attention Networks},
    booktitle = {European Computer Vision Conference},
    year      = {2018},
}
</code></pre>

# References
[1] Danelljan, M., Bhat, G., Khan, F.S., Felsberg, M.: ECO: Efficient convolution operators for tracking. In: CVPR (2017)

[2] Xiang, Y., Alahi, A., Savarese, S.: Learning to track: Online multi-object tracking by decision making. In: ICCV (2015)
