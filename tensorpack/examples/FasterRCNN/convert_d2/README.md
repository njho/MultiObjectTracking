## Detectron2 Conversion

We provide a script to migrate a model trained in
[detectron2](https://github.com/facebookresearch/detectron2)
to tensorpack.

The script reads a detectron2 config file and a detectron2 pkl file in the model zoo.
It produces a corresponding tensorpack configs, as well as a tensorpack-compatible checkpoint.

It currently supports ResNet{50,101}-{C4,FPN}-{Faster,Mask,Cascade} R-CNN models in
[detectron2 model zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md).

### Usage:

```
# 1. Download the corresponding model from detectron2 model zoo
# 2. Convert:

$ python convert_d2.py --d2-config detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --d2-pkl model_final_f10217.pkl --output R50FPN-d2-converted.npz
# the script will print tensorpack configs
'MODE_MASK=True' 'MODE_FPN=True' 'BACKBONE.STRIDE_1X1=True' 'PREPROC.PIXEL_MEAN=[123.675,116.28,103.53]' 'PREPROC.PIXEL_STD=[1.0,1.0,1.0]'

# 3. Use the above configs to verify the conversion is correct:
$ ./predict.py --evaluate out.json --load R50FPN-d2-converted.npz  --config DATA.BASEDIR=~/data/coco 'MODE_MASK=True' 'MODE_FPN=True' 'BACKBONE.STRIDE_1X1=True' 'PREPROC.PIXEL_MEAN=[123.675,116.28,103.53]' 'PREPROC.PIXEL_STD=[1.0,1.0,1.0]'

# 4. Naively convert the model to a frozen pb file:
$ ./predict.py --output-pb out.pb --load R50FPN-d2-converted.npz  --config DATA.BASEDIR=~/data/coco 'MODE_MASK=True' 'MODE_FPN=True' 'BACKBONE.STRIDE_1X1=True' 'PREPROC.PIXEL_MEAN=[123.675,116.28,103.53]' 'PREPROC.PIXEL_STD=[1.0,1.0,1.0]'
```

Note: this script does not support arbitrary detectron2 config.
When run against an unsupported config, it may fail silently and produce
erroneous models.

For models in detectron2 model zoo, there is a small incompatibility:

* `POOLER_SAMPLING_RATIO=0` in RoIAlign: there is no equivalence in TensorFlow.
	Our RoIAlign only implements `POOLER_SAMPLING_RATIO=2`.
	The results are quite similar, and the final AP may be different by <0.5.
