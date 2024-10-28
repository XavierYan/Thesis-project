#!/usr/bin/python
import pprint
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
# from vectornet_imitation import VectorNet,MultiAgentVectorNetActor
from backbone.vectornet import VectorNet, MultiAgentVectorNetActor

from dataset_with_feature import FeatureDataset
import logging
import os
import time
import random
from torchsummary import summary
from torchinfo import summary
import matplotlib.pyplot as plt
from datetime import datetime


import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import numpy as np
import random
import pprint
from datetime import datetime



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_logger(cfg):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(cfg['log_file'], mode='w')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def test(model,cfg, test_loader, device, logger):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            # prepare data
            ego_trajectory_batch = batch['history_trajectory'].to(device, dtype=torch.float)
            ego_reference_batch = batch['reference_trajectory'].to(device, dtype=torch.float)
            neighbour_trajectory_batch = batch['nearby_trajectories'].to(device, dtype=torch.float)
            neighbour_reference_batch = batch['nearby_reference_trajectories'].to(device, dtype=torch.float)
            vectormap_batch = batch['lane_boundaries'].to(device, dtype=torch.float)
            labels = batch['label'].to(device, dtype=torch.float)

            # predict
            predictions = model(ego_trajectory_batch, ego_reference_batch, neighbour_trajectory_batch, neighbour_reference_batch, vectormap_batch)

            # calulate loss
            loss = model.custom_loss(predictions, labels)
            loss_per_sample = loss / cfg['batch_size']
            total_loss += loss_per_sample.item()

    avg_loss = total_loss / len(test_loader)
    model.train()
    return avg_loss

def do_train(model, cfg, train_loader, test_records, optimizer, scheduler, writer_train, writer_test, logger):
    start_time = time.strftime('%Y-%m-%d %X', time.localtime(time.time()))
    device = cfg['device']
    print_every = cfg['print_every']
    epoch_losses = []
    for e in range(cfg['epochs']):
        progress_bar = tqdm(train_loader, desc=f'Epoch {e+1}/{cfg["epochs"]}', unit='batch')
        epoch_loss = 0
        for i, batch in enumerate(progress_bar):
            ego_trajectory_batch = batch['history_trajectory'].to(device, dtype=torch.float)
            ego_reference_batch = batch['reference_trajectory'].to(device, dtype=torch.float)
            neighbour_trajectory_batch = batch['nearby_trajectories'].to(device, dtype=torch.float)
            neighbour_reference_batch = batch['nearby_reference_trajectories'].to(device, dtype=torch.float)
            vectormap_batch = batch['lane_boundaries'].to(device, dtype=torch.float)
            labels = batch['label'].to(device, dtype=torch.float)

            optimizer.zero_grad()
            prediction = model(ego_trajectory_batch, ego_reference_batch, neighbour_trajectory_batch, neighbour_reference_batch, vectormap_batch)
            loss_per_sample = model.custom_loss(prediction, labels)
            loss_per_sample = loss_per_sample / cfg['batch_size']
            loss_per_sample.backward()
            optimizer.step()

            epoch_loss += loss_per_sample.item()

            # 计算并记录梯度范数
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    writer_train.add_scalar(f'Gradient/{name}', grad_norm, global_step=e * len(train_loader) + i)

            if i % print_every == 0:
                logger.info(f'Epoch {e+1}/{cfg["epochs"]}: Iteration {i}, loss = {loss_per_sample.item()}')

        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        writer_train.add_scalar('Loss/train', avg_loss, e)

        scheduler.step()

        if (e+1) % cfg['save_every'] == 0:
            file_path = os.path.join(cfg['save_path'], f"model_epoch{e+1}.pth")
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, file_path)
            logger.info(f"Saved model to {file_path}")

    torch.save(model.state_dict(), os.path.join(cfg['save_path'], "model_final.pth"))
    logger.info(f"Saved final model to {os.path.join(cfg['save_path'], 'model_final.pth')}")
    logger.info("Finish Training")
    end_time = time.strftime('%Y-%m-%d %X', time.localtime(time.time()))
    print(f'Start time -> {start_time}')
    print(f'End time -> {end_time}')


def main():
    setup_seed(20)
    USE_GPU = torch.cuda.is_available()
    device = torch.device('cuda' if USE_GPU else 'cpu')

    cfg = dict(
        device=device,
        learning_rate=1e-4,
        learning_rate_decay=0.3,
        epochs=100,
        batch_size=512,
        print_every=100,
        save_every=5,
        data_locate="generated_features/train/0405_0527.features.pkl",  #training feature path
        test_data_locate="generated_features/test/07_0328.features.pkl",  #testing feature path
        save_path="src/model_ckpt/0627/",
        log_file="src/log.txt",
        pretrained_weights_path=None, #weather use pretrained model
        use_pretrained=False
    )

    if not os.path.isdir(cfg['save_path']):
        os.makedirs(cfg['save_path'])
    
    TIMESTAMP = f"{datetime.now():%Y-%m-%dT%H-%M-%S}"
    train_log_dir = os.path.join('src/logs/train', TIMESTAMP)
    test_log_dir = os.path.join('src/logs/test', TIMESTAMP)
    writer_train = SummaryWriter(train_log_dir)
    writer_test = SummaryWriter(test_log_dir)

    logger = init_logger(cfg)

    train_dataset = FeatureDataset(cfg['data_locate'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=8, drop_last=True)

    test_dataset = FeatureDataset(cfg['test_data_locate'])
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=8)

    # model = VectorNet(cfg=cfg).to(device)
    vec = VectorNet()
    model = MultiAgentVectorNetActor( network = vec).to(device)
    # print(summary(model, ((1,50, 9), (1,50, 5), (1,4, 50, 9), (1,4, 50, 5), (1,4, 19, 8))))

    if cfg['use_pretrained']:
        # check pretrained weighs
        pretrained_weights_path = cfg.get('pretrained_weights_path', None)
        if pretrained_weights_path and os.path.isfile(pretrained_weights_path):
            checkpoint = torch.load(pretrained_weights_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded pretrained weights from {pretrained_weights_path}")
        else:
            logger.warning("Pretrained weights path is not provided or the file does not exist.")


    model.train()

    optimizer = optim.Adadelta(model.parameters(), rho=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=cfg['learning_rate_decay'])

    logger.info("Start Training...")
    do_train(model, cfg, train_loader, test_loader, optimizer, scheduler, writer_train, writer_test, logger)

if __name__ == "__main__":
    main()
