import tensorflow as tf


hparams = tf.contrib.training.HParams(
    # Audio:
    num_mels=80,
    n_fft=2048,
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,

    # train
    lr=0.001,
    train_steps=1000000,
    save_model_every=4000,
    gen_test_wave_every=10000,
    gen_file='./data/mels/LJ001-0001.mel',
    logdir_root='./logdir',
    decay_steps=50000,
    sigma=0.707,

    # network
    sample_size=25600,
    batch_size=1,
    upsampling_rate=256,  # same as hop_length
    n_flows=12,
    n_group=8,
    n_early_every=4,
    n_early_size=2,

    # local condition conv1d
    lc_conv1d=True,
    lc_conv1d_layers=2,
    lc_conv1d_filter_size=5,
    lc_conv1d_filter_num=128,

    # local condition encoding
    lc_encode=False,
    lc_encode_layers=2,
    lc_encode_size=128,

    # upsampling by transposed conv
    transposed_upsampling=False,
    transposed_conv_layers=2,
    transposed_conv_layer1_stride=16,
    transposed_conv_layer2_stride=16,
    transposed_conv_layer1_filter_width=16*5,  # filter width greater than stride, then could leverage context lc
    transposed_conv_layer2_filter_width=16*5,
    transposed_conv_channels=128,

    # wavenet
    n_layers=8,
    residual_channels=256,
    skip_channels=256,
    kernel_size=3,
)
