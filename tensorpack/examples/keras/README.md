
## Keras + Tensorpack

Use Keras to define a model and train it with efficient tensorpack trainers.

### Why?
Keras alone has various overhead. In particular, it is not efficient with large models.
The article [Towards Efficient Multi-GPU Training in Keras with TensorFlow](https://medium.com/rossum/towards-efficient-multi-gpu-training-in-keras-with-tensorflow-8a0091074fb2)
has mentioned some of it.

Even on a single GPU, tensorpack can run [1.2~2x faster](https://github.com/tensorpack/benchmarks/tree/master/other-wrappers)
than the equivalent Keras code. The gap becomes larger when you scale to multiple GPUs.
Tensorpack and [horovod](https://github.com/uber/horovod/blob/master/examples/keras_imagenet_resnet50.py)
are the only two tools I know that can scale the training of a large Keras model.

### Simple Examples:

There are two flavors where you can use a Keras model inside tensorpack:

1. Write the tower function similar to a standard tensorpack program, but mix some Keras layers in
	 between. See [mnist-keras.py](mnist-keras.py) on how to do this.
	 It does not support all tensorpack trainers, and can be brittle due to
	 incompatibilities between Keras and tensorpack.

2. The entire model to train is a Keras model (and there will be no `ModelDesc`, etc).
	 See [mnist-keras-v2.py](mnist-keras-v2.py).

### ImageNet Example:

[imagenet-resnet-keras.py](imagenet-resnet-keras.py):
reproduce exactly the same setting of [tensorpack ResNet example](../ResNet) on ImageNet.
It has:

+ ResNet-50 model modified from [keras.applications](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/_impl/keras/applications/resnet50.py).
	(We put stride on 3x3 conv in each bottleneck, which is different from certain other implementations).
+ Multi-GPU data-parallel __training and validation__ which scales
	+ Finished 100 epochs in 19 hours on 8 V100s, with >90% GPU utilization.
	+ Still slightly slower than native tensorpack examples.
+ Good accuracy (same as [tensorpack ResNet example](../ResNet))

### Note:

Keras does not respect variable scopes or variable
collections, which contradicts with tensorpack trainers.
Therefore Keras support is __experimental__ and __unofficial__.

These simple examples can run within tensorpack smoothly, but note that a
complicated model or a future version of Keras may break them.
