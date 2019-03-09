# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import librosa
from audio_utils import melspectrogram
import argparse
from params import hparams
import random
import codecs


def extract_melspectrum(wave_file, save_path, sr):
    try:
        y, _ = librosa.load(wave_file, sr=sr, mono=True)
        mel_spec = melspectrogram(y)
        data = np.array(mel_spec, 'float32')
        fid = open(save_path, 'wb')
        data.tofile(fid)
        fid.close()
        return save_path
    except Exception:
        raise


def extract_mfcc(wave_file, save_path, sr, n_mfcc):
    try:
        y, sr = librosa.load(wave_file, sr=sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = mfcc.T
        data = np.array(mfcc, 'float32')
        fid = open(save_path, 'wb')
        data.tofile(fid)
        fid.close()
        return save_path
    except Exception:
        raise


def gen_filelist(filelist, save_dir):
    random.shuffle(filelist)
    # random select 200 ids as test
    test_set = filelist[:200]
    train_set = filelist[200:]

    train_set.sort()
    test_set.sort()

    train_filelist = os.path.join(save_dir, 'train.scp')
    test_filelist = os.path.join(save_dir, 'test.scp')

    with codecs.open(train_filelist, 'w', 'utf-8') as f:
        for line in train_set:
            f.write(line)
            f.write('\n')

    with codecs.open(test_filelist, 'w', 'utf-8') as f:
        for line in test_set:
            f.write(line)
            f.write('\n')


def main(args):
    wave_files = glob.glob(os.path.join(args.wave_dir, '*.wav'))
    p = Pool(mp.cpu_count())

    results = []
    filelist = []
    for f in wave_files:
        f_name = os.path.basename(f)
        f_name = f_name[:-4]
        filelist.append(f_name)

        mel_save_path = os.path.join(args.mel_dir, f_name + '.mel')
        res = p.apply_async(extract_melspectrum, (f, mel_save_path, hparams.sample_rate))
        results.append(res)

    p.close()
    p.join()
    for res in results:
        output = res.get()
        print(output)

    gen_filelist(filelist, args.data_dir)

    print("job done!")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave_dir', type=str, help='wave directory')
    parser.add_argument('--mel_dir', type=str, default=None,
                        help='mel spectrum directory where to save the spectrum')
    parser.add_argument('--data_dir', type=str,
                        help='root folder of the data, where to save the filelist')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()
    if not os.path.exists(args.mel_dir):
        os.makedirs(args.mel_dir)

    main(args)
