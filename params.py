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
    lr=0.0001,
    train_steps=1000000,
    save_model_every=4000,
    logdir_root='./logdir',
    decay_steps=100000,
    sigma=0.707,

    # network
    sample_size=64000,
    batch_size=1,
    upsampling_rate=256,  # same as hop_length
    n_flows=12,
    n_group=8,
    n_early_every=4,
    n_early_size=2,

    # wavenet
    n_layers=8,
    residual_channels=256,
    skip_channels=256,
    kernel_size=3,
)
