import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from data_reader import read_binary_lc
import argparse
import os
from params import hparams
from glow import WaveGlow


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='Parallel WaveNet Network')
    parser.add_argument('--lc', type=str, default=None, required=True,
                        help='local condition file')
    parser.add_argument('--wave_name', type=str, default='waveglow.wav')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='restore model from checkpoint')
    parser.add_argument('--sigma', type=float, default=0.6,
                        help='sigma value for inference')
    return parser.parse_args()


def write_wav(waveform, sample_rate, filename):
    """

    :param waveform: [-1,1]
    :param sample_rate:
    :param filename:
    :return:
    """
    # TODO: write wave to 16bit PCM, don't use librosa to write wave
    y = np.array(waveform, dtype=np.float32)
    y *= 32767
    wavfile.write(filename, sample_rate, y.astype(np.int16))
    print('Updated wav file at {}'.format(filename))


def main():
    try:
        args = get_arguments()

        lc = read_binary_lc(args.lc, hparams.num_mels)
        # upsampling local condition
        lc = np.tile(lc, [1, 1, hparams.upsampling_rate])
        lc = np.reshape(lc, [1, -1, hparams.num_mels])
        print(lc.shape)

        glow = WaveGlow(lc_dim=hparams.num_mels,
                        n_flows=hparams.n_flows,
                        n_group=hparams.n_group,
                        n_early_every=hparams.n_early_every,
                        n_early_size=hparams.n_early_size)

        lc_placeholder = tf.placeholder(tf.float32, shape=[1, None, hparams.num_mels], name='lc')
        audio = glow.infer(lc_placeholder, sigma=args.sigma)

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
        print("restore model")
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.restore(sess, args.restore_from)
        print('restore model successfully!')

        audio_output = sess.run(audio, feed_dict={lc_placeholder: lc})
        audio_output = audio_output.flatten()
        print(audio_output)
        write_wav(audio_output, hparams.sample_rate, args.wave_name)
    except Exception:
        raise


if __name__ == '__main__':
    main()
