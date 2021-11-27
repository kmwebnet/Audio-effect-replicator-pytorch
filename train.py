import os
from argparse import ArgumentParser
import datetime
import yaml
from fx_replicator import (
    load_wave, flow, AverageMeter, EarlyStopping, Replicator
)
import numpy as np
import tqdm
import torch
import torchinfo
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
import adabound

def main():

    args = parse_args()

    with open(args.config_file) as fp:
        config = yaml.safe_load(fp)
    
    input_timesteps = config["input_timesteps"]
    output_timesteps = config["output_timesteps"]
    batch_size = config["batch_size"]
    max_epochs = config["max_epochs"]
    patience = config["patience"]

    train_dataset = [
        (load_wave(_[0]).reshape(-1, 1), load_wave(_[1]).reshape(-1, 1))
        for _ in config["train_data"]]
    train_dataflow = flow(train_dataset, input_timesteps, batch_size)

    val_dataset = [
        (load_wave(_[0]).reshape(-1, 1), load_wave(_[1]).reshape(-1, 1))
        for _ in config["val_data"]]
    val_dataflow = flow(val_dataset, input_timesteps, batch_size)

    timestamp = datetime.datetime.now()

    model = Replicator()

    loss_function = nn.MSELoss()
    optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.5)
    #optimizer = optim.Adam(model.parameters())

    # For tensorboard
    tb_log_dir = "tensorboard/{:%Y%m%d_%H%M%S}".format(timestamp)
    if not os.path.exists(tb_log_dir):
        os.makedirs(tb_log_dir)
    tb_writer = SummaryWriter(tb_log_dir)

    tloop = tqdm.trange(1, max_epochs + 1)
    es = EarlyStopping(patience=patience)

    steps = 100
    validation_time = 20


    cp_dir = "checkpoint/{:%Y%m%d_%H%M%S}".format(timestamp)
    if not os.path.exists(cp_dir):
        os.makedirs(cp_dir)

    idx = 0

    torchinfo.summary(model)

    for i in tloop:

        # TRAINING
        model.train() 
        tloop.set_description("Training Epoch")
        st = tqdm.trange( 0, steps )
        losses = AverageMeter()
        for j in st:

            xx ,yy  = train_dataflow.__next__()

            yz = torch.from_numpy(yy.astype(np.float32))
            st.set_description("Training Steps")
            optimizer.zero_grad()
            out1 = model(xx.astype(np.float32))

            loss = loss_function(out1[:, -output_timesteps:, -1:], yz[:, -output_timesteps:, :])

            losses.update(loss.detach().numpy())
            st.set_postfix(
                train_loss=losses.avg
            )

            loss.backward()
            optimizer.step()

            # write train graph
            tb_writer.add_scalar('train/loss', losses.avg, global_step=idx)
            idx += 1

        # VALIDATION
        model.eval() 
        vlosses = AverageMeter()
        for j in range(validation_time):
            vx ,vy  = val_dataflow.__next__()
            vyz = torch.from_numpy(vy.astype(np.float32))

            out2 = model(vx.astype(np.float32))

            #print(out2[:, -output_timesteps:, :])

            vloss = loss_function(out2[:, -output_timesteps:, -1:], vyz[:, -output_timesteps:, :])
            vlosses.update(vloss.detach().numpy())

        tloop.set_postfix(
            train_loss=losses.avg, val_loss=vlosses.avg
        )

        tb_writer.add_scalar('test/loss', vlosses.avg, global_step=i)

        stop = es.step(vlosses.avg)
        is_best = vlosses.avg == es.best

        if is_best:
            torch.save(model.state_dict(), os.path.join(
                cp_dir ,'best_result.pth'))
            print("\nbest updated")
        
        if stop:
            print("\nApply Early Stopping")
            break




def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file", "-c", default="./config.yml",
        help="configuration file (*.yml)")
    return parser.parse_args()

if __name__ == '__main__':
    main()
