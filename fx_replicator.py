import os
import datetime
import wave
import numpy as np
from numpy.lib.stride_tricks import as_strided
import torch
import torch.nn as nn
torch.manual_seed(1)

def load_wave(wave_file: str) -> np.float32:
    with wave.open(wave_file, "r") as w:
        buf = np.frombuffer(w.readframes(w.getnframes()) , dtype=np.int32) # type:ignore
    return (buf / 0x7fffffff).astype(np.float32)# type:ignore

def save_wave(buf, wave_file:str):
    _buf = (buf * 0x7fffffff).astype(np.int32)# type:ignore
    with wave.open(wave_file, "w") as w:
        w.setparams((1, 4, 48000, len(_buf), "NONE", "not compressed"))
        w.writeframes(_buf)

def flow(dataset, timesteps, batch_size):
    n_data = len(dataset)
    while True:
        i = np.random.randint(n_data)
        x, y = dataset[i]
        yield random_clop(x, y, timesteps, batch_size)

def random_clop(x, y, timesteps, batch_size):
    max_offset = len(x) - timesteps
    offsets = np.random.randint(max_offset, size=batch_size)
    batch_x = np.stack([x[offset:offset+timesteps] for offset in offsets])
    batch_y = np.stack([y[offset:offset+timesteps] for offset in offsets])
    return batch_x, batch_y

class Replicator(nn.Module):
    def __init__(self) -> None:
        super(Replicator,self).__init__()
        self.lstm0 = nn.LSTM(input_size=1, hidden_size=6, batch_first=True, num_layers=2)
      
    def forward(self,x):
        tx = torch.from_numpy(x.astype(np.float32)).clone()
        tx,_ = self.lstm0(tx)
        return tx
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1
            print("\n")
            print("bad_epoch:", self.num_bad_epochs)
            print("patience:", self.patience)

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

def sliding_window(x, window, slide):
    n_slide = (len(x) - window) // slide
    remain = (len(x) - window) % slide
    clopped = x[:-remain]
    return as_strided(clopped, shape=(n_slide + 1, window), strides=(slide * 4, 4))

