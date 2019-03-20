# WaveGlow
Tensorflow Implementation of [WaveGlow](https://arxiv.org/abs/1811.00002)

# How to run it
## step1: process data
process data by **preprocess_data.py**, following the command:
```
python preprocess_data.py --wave_dir=corpus\wavs --mel_dir=corpus\mels --data_dir=corpus\
```

## step2: train model
```
python train.py --filelist=xxx --wave_dir=xxx --lc_dir=xxx
```

model parameters are in file params.py


# TODO
* add transposed convolution for local condition upsamling
* add bi-directional local condition encoding
* Need to verify what's wrong in my usage of tf.nn.conv2d()

# Issues
## tf.nn.conv2d for dilated convlution does not covergence
In my first implementation of WaveGlow, I used **tf.nn.conv2d** to do dilated convolutions, the 3D Tensor(B\*T\*depth) is reshaped to 4D Tensor (B\*1\*T\*depth), and then leverage **tf.nn.conv2d** to do dilated convolution, but after many experiments I found that **tf.nn.conv2d** with dilated convolution does not convergence as expected. **For a long time, I have suspected that there maybe a bug in my implementation.**
* with a learning_rate=0.0001, the model does not convergence even after 652K steps.
* with a learning_rate=0.001, the model does not convergence even after 552K steps.

Example waves by **tf.nn.conv2d** are in samples/tf_conv2d_as_dilated_conv

In implementation [b04901014/waveglow-tensorflow](https://github.com/b04901014/waveglow-tensorflow), the author also used **tf.nn.conv2d** for dilated convolution, this code convergence but **very very slow**. So there maybe something wrong in my usage.

## private dilated 1D convolution 
**tf.nn.conv2d()** for dilated convolution did not convergence as expected in my experiments, so I changed the dilated convolution to implementation from [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet).

