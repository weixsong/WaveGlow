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

# Issues
* tf.nn.conv2d() for dilated convolution did not convergence as expected in my experiments, so I changed the dilated convolution to implementation from [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet).
* Need to verify what's wrong in my usage of tf.nn.conv2d()
* **tf.nn.conv2d()** for dilated convolution is not supported on tensorflow CPU implementation, so not easy to debug on CPU machine.
