

import hparams
import transformers
import os
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
from torch.utils.data import DataLoader
# import torchvision.utils as vutils
from dataset import MyDataset
from TransformerTTSModel import LSTransformer
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import argparse
from collections import OrderedDict
import plot
import numpy as np
import matplotlib.pyplot as plt
import audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', type=bool, help='Global step to restore checkpoint', default=False)
    parser.add_argument('--restore_step', type=int, help='Global step to restore checkpoint', default=3000)
    args = parser.parse_args()
    print("Reading data parameters...")
    model = LSTransformer()
    model = model.cuda()
    now = datetime.now()
    num = len(next(os.walk('/content/Transformer-LST/runs'))[1]) + 1
    writer = SummaryWriter("/content/Transformer-LST/runs/{}_{}".format(now.strftime("%Y-%m-%d_%H-%M-%S"), num))
    # net = nn.DataParallel(model).cuda()
    torch.manual_seed(hparams.random_seed)
    torch.cuda.manual_seed_all(hparams.random_seed)

    dataset = MyDataset(hparams.data_root, 'train')
    transforms = [
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35)
    ]
    optimizer = transformers.AdamW(model.parameters(), lr=hparams.lr)
    epoch_start = 1
    model_train_loss = []
    model_eval_loss = []
    next_mel_input = None

    if args.restore:
        print("Restoring Checkpoint...")
        # model.load_state_dict(load_checkpoint("transformer", args.restore_step))
        state_dict = torch.load('./logs/checkpoint/checkpoint_%s_%d.pth.tar' % ("transformer", args.restore_step), map_location='cpu')
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = state_dict['epoch'] + 1
        model = model.cuda()
        if epoch_start > 20:
            next_mel_input  = state_dict['pred_mel']

    loader = load_data_to_dataloader(dataset)

    print('num_train_data:{}'.format(len(dataset.data)))
    # print("len(dataset)", len(dataset))
    num_training_steps = hparams.epochs * len(dataset) // hparams.batch_size

    # scheduler = transformers.get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=hparams.warmup_steps * num_training_steps,
    #     num_training_steps=num_training_steps
    # )
    best_loss = 1e10
    if args.restore:
        print(f'---------[INFO] Restarting Training from Epoch {epoch_start} -----------\n')