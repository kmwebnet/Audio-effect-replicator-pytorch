from argparse import ArgumentParser
from time import time
import yaml
import numpy as np
from fx_replicator import (
    load_wave, save_wave, sliding_window, Replicator
)
import torch
import torchinfo

def main():

    args = parse_args()

    with open(args.config_file) as fp:
        config = yaml.safe_load(fp)

    input_timesteps = config["input_timesteps"]
    output_timesteps = config["output_timesteps"]
    batch_size = config["batch_size"]

    data = load_wave(args.input_file)
    print("data.shape is:", data.shape)
    print("data.len is:", len(data)) #type:ignore

    # padding and rounded up to the batch multiple
    block_size = output_timesteps * batch_size
    prepad = input_timesteps - output_timesteps
    postpad = len(data) % block_size #type:ignore
    print("postpad", block_size - postpad)
    padded = np.concatenate(( #type:ignore
        np.zeros(prepad, np.float32), #type:ignore
        data,
        np.zeros(block_size - postpad, np.float32))) #type:ignore
    x = sliding_window(padded, input_timesteps, output_timesteps)
    x = x[:, :, np.newaxis]
    y = torch.zeros_like(torch.from_numpy(x.astype(np.float32)))

    batchlen = x.shape[0]
    print("x.length is:",batchlen) 

    model = Replicator()
    torchinfo.summary(model)

    model.load_state_dict(torch.load(args.model_file))
    model.eval()

    xx = np.zeros((batch_size,input_timesteps,1),np.float32)

    print("xx.shape is:", xx.shape)

    print("x.shape in the loop is:", x[32:32 + batch_size , : , : ].shape)

    start1 = time()

    for step in range(0, batchlen , batch_size):

        xx = x[step:step + batch_size , : , : ]

        yy = model(xx.astype(np.float32))

        y[step:step + batch_size , : , : ] = yy[:,:,-1:]

        proc_time = time() - start1
        print(proc_time)
        print(step)
        
    y = y.to('cpu').detach().numpy()
    y = y[:, -output_timesteps:, :].reshape(-1)[:len(data)]

    save_wave(y, args.output_file)

    print("finished\n")
    proc_time = time() - start1
    print(proc_time)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file", "-c", default="./config.yml",
        help="configuration file (*.yml)")
    parser.add_argument(
        "--input_file", "-i",
        help="input wave file (48kHz/mono, *.wav)")
    parser.add_argument(
        "--output_file", "-o", default="./predicted.wav",
        help="output wave file (48kHz/mono, *.wav)")
    parser.add_argument(
        "--model_file", "-m",
        help="input model file (*.pth)")
    return parser.parse_args()

if __name__ == '__main__':
    main()
