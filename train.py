# -*- coding: utf-8 -*-
"""Train EAGG-UNet.

Usage::

    python train.py --config configs/train.json
    python train.py --config configs/train.json --save_name eagg_lcma
"""

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import eval_metrics
from models import build_model
from utils.unet_dataset import UnetDataset, build_transform


def select_device(config):
    """'auto' -> cuda:0 when available, otherwise CPU."""
    want = config.get('device', 'auto')
    if want in (None, 'auto', ''):
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(want)


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iou_to_string(iou):
    """Format an IoU array (or scalar) for logging."""
    if np.isscalar(iou):
        return '{:.4f}'.format(iou)
    body = ', '.join('{}: {:.4f}'.format(i, v) for i, v in enumerate(iou))
    return '{' + body + '}'


def init_logger(model_name, log_dir='logs'):
    """Fresh file logger per run."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    os.makedirs(log_dir, exist_ok=True)

    log_name = os.path.join(log_dir, '{}_{}.log'.format(model_name, rq))

    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def train(config, save_name=None):
    """Train one EAGG-UNet model.

    Args:
        config: configuration dict (see ``configs/train.json``).
        save_name: checkpoint prefix; defaults to ``config['model_name']``.

    Returns:
        Best validation mIoU (float).
    """
    # ---------- setup ----------
    device = select_device(config)
    set_seed(config.get('seed'))

    save_name = save_name or config.get('model_name', 'EAGG')

    logger = init_logger(save_name, config.get('log_dir', 'logs'))
    logger.info('=' * 60)
    logger.info('model: {} | device: {}'.format(save_name, device))
    logger.info('=' * 60)

    num_classes = config['num_classes']
    in_channels = config.get('in_channels', 5)

    logger.info('num_classes: {}'.format(num_classes))
    logger.info('in_channels: {}'.format(in_channels))

    # ---------- model ----------
    model = build_model(config).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('trainable parameters: {:,}'.format(total_params))
    logger.info('=' * 60)

    # ---------- loss ----------
    criterion = nn.CrossEntropyLoss()
    logger.info('loss: CrossEntropyLoss (ignore_index=-100)')

    # ---------- data ----------
    transform = build_transform()
    num_workers = config.get('num_workers', 0)

    dst_train = UnetDataset(config['train_list'], transform=transform, train=True)
    dataloader_train = DataLoader(
        dst_train,
        shuffle=True,
        batch_size=config['batch_size'],
        num_workers=num_workers,
    )

    dst_valid = UnetDataset(config['test_list'], transform=transform, train=False)
    dataloader_valid = DataLoader(
        dst_valid,
        shuffle=False,
        batch_size=config['batch_size'],
        num_workers=num_workers,
    )

    # ---------- optimizer ----------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        betas=[config['momentum'], 0.999],
        weight_decay=config['weight_decay'],
    )

    # ---------- bookkeeping ----------
    best_val_loss = float('inf')
    best_val_miou = 0.0
    best_val_oa = 0.0
    patience_counter = 0
    early_stop_patience = config.get('early_stop_patience', 30)
    min_epochs = config.get('min_epochs', 50)

    best_miou_epoch = 0
    best_miou_loss = float('inf')
    best_miou_oa = 0.0

    best_loss_epoch = 0
    best_loss_miou = 0.0
    best_loss_oa = 0.0

    save_path = config['save_model']['save_path']
    os.makedirs(save_path, exist_ok=True)

    logger.info('=' * 60)
    logger.info('start training: {}'.format(save_name))
    logger.info('   epochs: {}'.format(config['num_epoch']))
    logger.info('   batch size: {}'.format(config['batch_size']))
    logger.info('   lr: {}'.format(config['lr']))
    logger.info('   weight decay: {}'.format(config['weight_decay']))
    logger.info('   momentum: {}'.format(config['momentum']))
    logger.info('   early stop patience: {}'.format(early_stop_patience))
    logger.info('   min epochs: {}'.format(min_epochs))
    logger.info('=' * 60)

    for epoch in range(config['num_epoch']):
        epoch_start = time.time()

        # ===== train =====
        model.train()
        loss_sum = 0.0
        correct_sum = 0.0
        labeled_sum = 0.0
        inter_sum = 0.0
        unoin_sum = 0.0
        pixelAcc = 0.0
        IoU_array = np.zeros(num_classes)
        conf_matrix_train = np.zeros((num_classes, num_classes))
        tbar = tqdm(dataloader_train, ncols=120)

        for batch_idx, (data, target, _) in enumerate(tbar):
            tic = time.time()

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

            correct, labeled, inter, unoin, conf_matrix_train = eval_metrics(
                output, target, num_classes, conf_matrix_train
            )
            correct_sum += correct
            labeled_sum += labeled
            inter_sum += inter
            unoin_sum += unoin

            pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
            IoU_array = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)

            tbar.set_description(
                'TRAIN ({}) | Loss: {:.5f} | OA {:.5f} mIoU {:.5f} | bt {:.2f} et {:.2f}|'.format(
                    epoch, loss_sum / (batch_idx + 1), pixelAcc, IoU_array.mean(),
                    time.time() - tic, time.time() - epoch_start))

        train_loss = loss_sum / (batch_idx + 1)
        train_miou = IoU_array.mean()
        train_oa = pixelAcc

        logger.info('TRAIN ({}) | Loss: {:.5f} | OA {:.5f} IOU {}  mIoU {:.5f} '.format(
            epoch, train_loss, train_oa, iou_to_string(IoU_array), train_miou))

        # ===== validate =====
        test_start = time.time()

        model.eval()
        loss_sum = 0.0
        correct_sum = 0.0
        labeled_sum = 0.0
        inter_sum = 0.0
        unoin_sum = 0.0
        pixelAcc = 0.0
        mIoU_array = np.zeros(num_classes)
        tbar = tqdm(dataloader_valid, ncols=120)
        class_precision = np.zeros(num_classes)
        class_recall = np.zeros(num_classes)
        class_f1 = np.zeros(num_classes)
        conf_matrix_val = np.zeros((num_classes, num_classes))

        with torch.no_grad():
            for batch_idx, (data, target, _) in enumerate(tbar):
                tic = time.time()

                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = criterion(output, target)
                loss_sum += loss.item()

                correct, labeled, inter, unoin, conf_matrix_val = eval_metrics(
                    output, target, num_classes, conf_matrix_val
                )
                correct_sum += correct
                labeled_sum += labeled
                inter_sum += inter
                unoin_sum += unoin
                pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
                mIoU_array = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)

                tbar.set_description(
                    'VAL ({}) | Loss: {:.5f} | Acc {:.5f} mIoU {:.5f} | bt {:.2f} et {:.2f}|'.format(
                        epoch, loss_sum / (batch_idx + 1), pixelAcc, mIoU_array.mean(),
                        time.time() - tic, time.time() - test_start))

        val_loss = loss_sum / (batch_idx + 1)
        val_miou = mIoU_array.mean()
        val_oa = pixelAcc

        # per-class precision / recall / F1 from the accumulated matrix
        for i in range(num_classes):
            if conf_matrix_val[:, i].sum() > 0:
                class_precision[i] = conf_matrix_val[i, i] / conf_matrix_val[:, i].sum()
            if conf_matrix_val[i].sum() > 0:
                class_recall[i] = conf_matrix_val[i, i] / conf_matrix_val[i].sum()
            if (class_precision[i] + class_recall[i]) > 0:
                class_f1[i] = (2.0 * class_precision[i] * class_recall[i]) / (
                    class_precision[i] + class_recall[i])

        logger.info(
            'VAL ({}) | Loss: {:.5f} | OA {:.5f} |IOU {} |mIoU {:.5f} '
            '|class_precision {}| class_recall {} | class_f1 {}|'.format(
                epoch, val_loss, val_oa, iou_to_string(mIoU_array), val_miou,
                iou_to_string(class_precision), iou_to_string(class_recall),
                iou_to_string(class_f1)))

        # ===== best mIoU =====
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_miou_epoch = epoch
            best_miou_loss = val_loss
            best_miou_oa = val_oa

            torch.save(
                model.state_dict(),
                os.path.join(save_path, '{}_best_miou.pth'.format(save_name))
            )

            np.savetxt(
                os.path.join(save_path, '{}_best_miou.txt'.format(save_name)),
                np.array([epoch, val_loss, val_miou, val_oa])
            )

            np.savetxt(
                os.path.join(save_path, '{}_conf_matrix_val.txt'.format(save_name)),
                conf_matrix_val,
                fmt='%d'
            )

            logger.info('best mIoU model saved (mIoU: {:.4f}, OA: {:.4f}, epoch: {})'.format(
                val_miou, val_oa, epoch))
            print('best mIoU model saved (mIoU: {:.4f}, OA: {:.4f})'.format(val_miou, val_oa))

            patience_counter = 0
        else:
            patience_counter += 1

        # ===== best loss =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_epoch = epoch
            best_loss_miou = val_miou
            best_loss_oa = val_oa

            torch.save(
                model.state_dict(),
                os.path.join(save_path, '{}_best_loss.pth'.format(save_name))
            )

            np.savetxt(
                os.path.join(save_path, '{}_best_loss.txt'.format(save_name)),
                np.array([epoch, val_loss, val_miou, val_oa])
            )

        # ===== early stopping =====
        if epoch >= min_epochs and patience_counter >= early_stop_patience:
            logger.info('=' * 60)
            logger.info('early stopping triggered')
            logger.info('   best mIoU: {:.4f} (epoch {})'.format(best_val_miou, best_miou_epoch))
            logger.info('   no improvement for {} epochs'.format(patience_counter))
            logger.info('=' * 60)
            print('\nearly stopping! best mIoU: {:.4f} (epoch {})'.format(
                best_val_miou, best_miou_epoch))
            break

    # ===== summary =====
    logger.info('=' * 60)
    logger.info('training finished. best results:')
    logger.info('   model: {}'.format(save_name))
    logger.info('   [best mIoU] epoch: {} | mIoU: {:.4f} | OA: {:.4f} | loss: {:.4f}'.format(
        best_miou_epoch, best_val_miou, best_miou_oa, best_miou_loss))
    logger.info('   [best loss] epoch: {} | loss: {:.4f} | mIoU: {:.4f} | OA: {:.4f}'.format(
        best_loss_epoch, best_val_loss, best_loss_miou, best_loss_oa))
    logger.info('=' * 60)

    print('\ntraining finished. best mIoU: {:.4f} (epoch {})'.format(
        best_val_miou, best_miou_epoch))

    return best_val_miou


def main():
    parser = argparse.ArgumentParser(description='Train EAGG-UNet')
    parser.add_argument('--config', type=str, default='configs/train.json',
                        help='path to the training config json')
    parser.add_argument('--save_name', type=str, default=None,
                        help='checkpoint prefix (default: config["model_name"])')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = json.load(f)

    train(config, save_name=args.save_name)


if __name__ == '__main__':
    main()
