import os
import random
import threading
import codecs
import queue
import librosa
import numpy as np
from params import hparams


def read_binary_lc(file_path, dimension):
    f = open(file_path, 'rb')
    features = np.fromfile(f, dtype=np.float32)
    f.close()
    assert features.size % float(dimension) == 0.0,\
        'specified dimension %s not compatible with data' % (dimension,)
    features = features.reshape((-1, dimension))
    return features


def read_wave_and_lc_features(filelist_scpfile, wave_dir, lc_dir):
    filelist = []
    with codecs.open(filelist_scpfile, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            file_id = line
            filelist.append(file_id)

    random.shuffle(filelist)
    for file_id in filelist:
        wave_path = os.path.join(wave_dir, file_id + '.wav')
        lc_path = os.path.join(lc_dir, file_id + '.mel')

        # read wave
        audio, _ = librosa.load(wave_path, sr=hparams.sample_rate, mono=True)
        audio = audio.reshape(-1, 1)

        # read local condition
        lc_features = read_binary_lc(lc_path, hparams.num_mels)

        yield audio, lc_features, file_id


class DataReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 coord,
                 filelist,
                 wave_dir,
                 lc_dir,
                 queue_size=512):
        self.coord = coord
        self.filelist = filelist
        self.wave_dir = wave_dir
        self.lc_dir = lc_dir
        self.lc_dim = hparams.num_mels
        self.lc_frames = hparams.sample_size // hparams.upsampling_rate
        # recompute a sample size
        self.sample_size = self.lc_frames * hparams.upsampling_rate
        self.upsample_rate = hparams.upsampling_rate
        self.threads = []
        self.queue = queue.Queue(maxsize=queue_size)

    def dequeue(self, num_elements):
        batch_audio = np.empty([0, self.sample_size, 1])
        batch_lc = np.empty([0, self.lc_frames, self.lc_dim])
        for i in range(num_elements):
            audio, lc = self.queue.get(block=True)
            audio = np.reshape(audio, [1, self.sample_size, 1])
            lc = np.reshape(lc, [1, self.lc_frames, self.lc_dim])
            batch_audio = np.concatenate([batch_audio, audio], axis=0)
            batch_lc = np.concatenate([batch_lc, lc], axis=0)

        return batch_audio, batch_lc

    def thread_main(self):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = read_wave_and_lc_features(self.filelist,
                                                 self.wave_dir,
                                                 self.lc_dir)
            for audio, lc_features, file_id in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                # force align wave & local condition
                if len(audio) > len(lc_features) * self.upsample_rate:
                    # clip audio
                    audio = audio[:len(lc_features) * self.upsample_rate, :]
                elif len(audio) < len(lc_features) * self.upsample_rate:
                    # clip local condition and audio
                    audio_frames = len(audio) // self.upsample_rate
                    frames = min(audio_frames, len(lc_features))
                    audio = audio[:frames*self.upsample_rate, :]
                    lc_features = lc_features[:frames, :]
                else:
                    pass

                # TODO: add random-ness for the data-generator

                while len(audio) >= self.sample_size and len(lc_features) >= self.lc_frames:
                    audio_piece = audio[:self.sample_size, :]
                    lc_piece = lc_features[:self.lc_frames, :]
                    self.queue.put([audio_piece, lc_piece])

                    audio = audio[self.sample_size:, :]
                    lc_features = lc_features[self.lc_frames:, :]

    def start_threads(self, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=())
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)

        return self.threads
