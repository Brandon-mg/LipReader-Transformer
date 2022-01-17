from os.path import basename, join, dirname, isfile
import hparams
import numpy as np
import torch
import os
from glob import glob
from torch.utils.data import Dataset
import cv2
import torchaudio

def get_frame_id(frame):
    return int(basename(frame).split('.')[0])


def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder


class MyDataset(Dataset):
    def __init__(self, data_root, phase):
        self.phase = phase
        self.hparams = hparams
        self.window_fnames = []
        self.frame_name = None
        self._batches_per_group = 4
        self._target_pad = -hparams.max_abs_value
        self._pad = 0
        self.x = None
        self.mel = None
        self.data = []
        print(data_root)
        with open('./{}.txt'.format(phase)) as vidlist:
            for vid_id in vidlist:
                vid_id = vid_id.strip()
                self.data.extend(list(glob(os.path.join(data_root, 'preprocessed', vid_id, '*/*.jpg'))))

    def __getitem__(self, idx):
        window = self.get_example(idx)
        input_length = len(window[0])
        mel_length = len(window[1])
        mel_target = window[1]
        # print("input_length", input_length)
        input_data, input_max_len = self._prepare_inputs(window[0])
        # print("input data", input_data.shape)
        # pos_inp = np.arange(1, input_length + 1)
        # pos_mel = np.arange(1, mel_length + 1)
        mel_mask = [1] * (mel_target.shape[0]+1)

        end_logits = [0] * (mel_length - 1)
        end_logits += [1]
        start_token =  [-4.] * 1
        end_token =  [-5.] * 1
        empty_token = [-4.] * (mel_target.shape[0] - 1)
        # mel_target, mel_target_max_len, mel_max_len = self._prepare_targets(window[1], alignment=2)

        return [torch.FloatTensor(input_data), mel_target, torch.tensor(mel_mask, dtype=torch.long),
        torch.tensor(end_logits, dtype=torch.float), torch.tensor(start_token, dtype=torch.float),
        torch.tensor(end_token, dtype=torch.float), torch.tensor(empty_token, dtype=torch.float)]

    def __len__(self):
        return len(self.data)

    ''' Adapted from Lip2Wav (https://github.com/Rudrabha/Lip2Wav) '''

    def get_example(self, idx):
        self.frame_name = self.data[idx]
        self.window_fnames = self.get_window(self.frame_name)
        
        if self.window_fnames is None:
            idx = np.random.randint(len(self.data))
            self.get_example(idx)
        if len(self.window_fnames) != self.hparams.T:
           
            idx = np.random.randint(len(self.data))
            self.get_example(idx)

        self.mel = np.load(os.path.join(os.path.dirname(self.frame_name), 'mels.npz'))['spec'].T
        self.mel = self.crop_audio_window(self.mel, self.frame_name)

        if self.mel.shape[0] != self.hparams.mel_step_size:
            idx = np.random.randint(len(self.data))
            self.get_example(idx)
        # print("len self.window_fnames", len(self.window_fnames))
        window = []
        for fname in self.window_fnames:
            img = cv2.imread(fname)
            try:
                img = cv2.resize(img, (self.hparams.img_size, self.hparams.img_size))
            except:
                continue

            window.append(img)
        # print("window", len(window))
        self.x = np.asarray(window) / 255.

        return [self.x, self.mel, len(self.mel)]

    def get_window(self, center_frame):
        center_id = get_frame_id(center_frame)
        vidname = dirname(center_frame)
        if self.hparams.T % 2:
            window_ids = range(center_id - self.hparams.T // 2, center_id + self.hparams.T // 2 + 1)
        else:
            window_ids = range(center_id - self.hparams.T // 2, center_id + self.hparams.T // 2)

        self.window_fnames = []
        for frame_id in window_ids:
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            self.window_fnames.append(frame)

        return self.window_fnames

    def crop_audio_window(self, spec, center_frame):
        # estimate total number of frames from spec (num_features, T)
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_id = get_frame_id(center_frame) - self.hparams.T // 2
        total_num_frames = int((spec.shape[0] * self.hparams.hop_size * self.hparams.fps) / self.hparams.sample_rate)
        start_idx = int(spec.shape[0] * start_frame_id / float(total_num_frames))
        end_idx = start_idx + self.hparams.mel_step_size
        return spec[start_idx: end_idx, :]

    def _prepare_inputs(self, inputs):  # inputs shape: 90,96,96,3
        max_len = max([len(x) for x in inputs])  # (1,96,96,3),(2,96,96,3)......(90,96,96,3) therefore max=90
        return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len

    def _prepare_targets(self, targets, alignment):  # alignment = outputs per step = 2
        max_len = max([len(t) for t in targets])
        data_len = _round_up(max_len, alignment)
        return np.stack([self._pad_target(t, data_len) for t in targets]), data_len, max_len

    def _pad_input(self, x, length):
        return np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=self._pad)

    def _pad_target(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode="constant", constant_values=self._target_pad)

