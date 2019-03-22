# WaveGlow
Tensorflow Implementation of [WaveGlow](https://arxiv.org/abs/1811.00002)

# ATTENTION PLEASE
**ATTENTION**: I have verified that if you use **tf.nn.conv2d()** for dilated convolution with data format **NHWC**, the model does not convergence. I have tried at least more that 30 experiments.

Because **tf.nn.conv2d()** with data format **NHWC** does not convergence, so in master branch I changed the dilated convolution implementation to implementation in [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet).

In my experiment with **tf.nn.conv2d()** by data format **NHWC**, even after **652K** steps the model still did not convergence. See example in <code>./samples/tf_conv2d_as_dilated_conv/</code>

I spent lots of time to investigate what's wrong with my code by using **tf.nn.conv2d** with data format **NHWC**, after the private implemented dilated convolution convergences as expected, so I doubt there maybe a bug in Tensorflow's implementation for dilated convolution. 

And, I tried using **tf.nn.conv2d()** by data format **NCHW**, then the model convergences quickly as expected, see example in <code>samples/tf_conv2d_NCHW</code>, **so there is a bug in Tensorflow's dilated convolution with data format NHWC**.

# How to run it
## step1: process data
process data by **preprocess_data.py**, following the command:
```
python preprocess_data.py --wave_dir=corpus\wavs --mel_dir=corpus\mels --data_dir=corpus
```

## step2: train model
```
python train.py --filelist=xxx --wave_dir=xxx --lc_dir=xxx
```

model parameters are in file params.py


# TODO
* add transposed convolution for local condition upsamling
* add bi-directional local condition encoding

# Issues
## tf.nn.conv2d for dilated convlution does not covergence
In my first implementation of WaveGlow, I used **tf.nn.conv2d** to do dilated convolutions, the 3D Tensor(B\*T\*depth) is reshaped to 4D Tensor (B\*1\*T\*depth), and then leverage **tf.nn.conv2d** to do dilated convolution, but after many experiments I found that **tf.nn.conv2d** with dilated convolution does not convergence as expected. **For a long time, I have suspected that there maybe a bug in my implementation.**
* with a learning_rate=0.0001, the model does not convergence even after 652K steps.
* with a learning_rate=0.001, the model does not convergence even after 552K steps.

Example waves by **tf.nn.conv2d** are in samples/tf_conv2d_as_dilated_conv

In implementation [b04901014/waveglow-tensorflow](https://github.com/b04901014/waveglow-tensorflow), the author also used **tf.nn.conv2d** for dilated convolution, this code convergence but **very very slow**. So there maybe something wrong in my usage.

## private dilated 1D convolution 
**tf.nn.conv2d()** for dilated convolution did not convergence as expected in my experiments, so I changed the dilated convolution to implementation from [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet).

