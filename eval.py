# -*- coding: utf-8 -*-
"""Evaluate EAGG-UNet on a test list.

Usage::

    python eval.py --config configs/eval.json
    python eval.py --config configs/eval.json --checkpoint saved/eagg_best_miou.pth
"""

import argparse
import json
import os

import numpy as np
import torch

# Import the dataset helpers before anything that pulls in torchvision: GDAL
# must be loaded first, otherwise some conda environments fail to import it
# (libtiff / libjpeg symbol clash). See README "Troubleshooting".
from utils.unet_dataset import build_transform, read_label, read_tiff, to_tensor

from metrics import eval_metrics
from models import build_model


def load_state_dict(model, checkpoint_path):
    """Strict load with a shape-filtered fallback for mismatched checkpoints."""
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # unwrap a training checkpoint saved as {'state_dict': ...}
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    try:
        model.load_state_dict(state_dict, strict=True)
        print('weights loaded (strict)')
    except Exception as e:
        print('strict load failed: {}'.format(e))
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print('loaded {}/{} tensors'.format(len(filtered), len(model_dict)))

    return model


def evaluate(config, checkpoint_path=None):
    """Run evaluation over the list in ``config['img_txt']``.

    Returns:
        dict with ``OA``, ``mIoU`` and ``Macro_F1``.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_classes = config.get('num_classes', 20)
    model_name = config.get('model_name', 'EAGG')

    print('=' * 60)
    print('model: {} | device: {}'.format(model_name, device))
    print('=' * 60)

    # ---------- model ----------
    model = build_model(config)

    if checkpoint_path is None:
        checkpoint_path = config.get(
            'checkpoint',
            os.path.join(config['save_model']['save_path'],
                         '{}_best_miou.pth'.format(model_name))
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('checkpoint not found: {}'.format(checkpoint_path))

    print('checkpoint: {}'.format(checkpoint_path))
    load_state_dict(model, checkpoint_path)

    transform = build_transform()

    model.to(device)
    model.eval()

    # ---------- accumulators ----------
    conf_matrix_test = np.zeros((num_classes, num_classes))
    correct_sum = 0.0
    labeled_sum = 0.0
    inter_sum = np.zeros(num_classes)
    unoin_sum = np.zeros(num_classes)
    class_precision = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_f1 = np.zeros(num_classes)

    txt_path = config.get('img_txt')
    if txt_path is None:
        raise ValueError("config is missing the 'img_txt' field")
    if not os.path.exists(txt_path):
        raise FileNotFoundError('data list not found: {}'.format(txt_path))

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    base_dir = os.path.dirname(os.path.abspath(txt_path))
    print('samples in list: {}'.format(len(lines)))

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        image_name, label_name = parts[0], parts[1]

        if not os.path.isabs(image_name):
            image_name = os.path.join(base_dir, image_name)
        if not os.path.isabs(label_name):
            label_name = os.path.join(base_dir, label_name)

        if not os.path.exists(image_name):
            print('missing image: {}'.format(image_name))
            continue
        if not os.path.exists(label_name):
            print('missing label: {}'.format(label_name))
            continue

        try:
            label = read_label(label_name)
            label = torch.from_numpy(np.asarray(label, dtype=np.int32)).long().unsqueeze(0)
        except Exception as e:
            print('failed to read label {}: {}'.format(label_name, e))
            continue

        try:
            image = read_tiff(image_name, train=True)
            image = np.array(image)
            image = np.transpose(image, (1, 2, 0))
            image = to_tensor(image)
            image = image.to(torch.float32)
            image = transform(image)
            image = image.unsqueeze(0)
        except Exception as e:
            print('failed to read image {}: {}'.format(image_name, e))
            continue

        image = image.to(device)
        label = label.to(device)

        with torch.no_grad():
            output = model(image)

        correct, labeled, inter, unoin, conf_matrix_test = eval_metrics(
            output, label, num_classes, conf_matrix_test
        )
        correct_sum += correct
        labeled_sum += labeled
        inter_sum = inter_sum + inter
        unoin_sum = unoin_sum + unoin

        iou_per_class = inter_sum / (np.spacing(1) + unoin_sum)
        mean_iou = iou_per_class.mean()

        if (idx + 1) % 10 == 0:
            pixelAcc = correct_sum / (np.spacing(1) + labeled_sum)
            print('  progress: {}/{}, OA: {:.4f}, mIoU: {:.4f}'.format(
                idx + 1, len(lines), pixelAcc, mean_iou))

    # ---------- final metrics ----------
    pixelAcc = correct_sum / (np.spacing(1) + labeled_sum)
    mean_iou = iou_per_class.mean()

    for i in range(num_classes):
        if conf_matrix_test[:, i].sum() > 0:
            class_precision[i] = conf_matrix_test[i, i] / conf_matrix_test[:, i].sum()
        if conf_matrix_test[i].sum() > 0:
            class_recall[i] = conf_matrix_test[i, i] / conf_matrix_test[i].sum()
        if (class_precision[i] + class_recall[i]) > 0:
            class_f1[i] = (2.0 * class_precision[i] * class_recall[i]) / (
                class_precision[i] + class_recall[i])

    macro_f1 = np.mean(class_f1)

    print('=' * 60)
    print('per-class metrics:')
    for i in range(num_classes):
        print('class {:2d}: P={:.3f} R={:.3f} F1={:.3f} support={}'.format(
            i, class_precision[i], class_recall[i], class_f1[i],
            conf_matrix_test[i].sum()))

    valid_mask = conf_matrix_test.sum(axis=1) > 0
    if valid_mask.sum() > 0:
        print('macro F1 over present classes: {:.4f}'.format(
            class_f1[valid_mask].mean()))

    print('=' * 60)
    print('results ({})'.format(model_name))
    print('=' * 60)
    print('checkpoint: {}'.format(checkpoint_path))
    print('-' * 60)
    print('  OA (Overall Accuracy):  {:.5f}'.format(pixelAcc))
    print('  mIoU (Mean IoU):        {:.5f}'.format(mean_iou))
    print('  Macro F1:               {:.5f}'.format(macro_f1))
    print('=' * 60)

    # ---------- persist ----------
    save_dir = config.get('result_dir', 'confuse_matrix')
    os.makedirs(save_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    matrix_path = os.path.join(save_dir, '{}_matrix.txt'.format(base_name))
    result_path = os.path.join(save_dir, '{}_results.txt'.format(base_name))

    np.savetxt(matrix_path, conf_matrix_test, fmt='%d')
    print('confusion matrix saved: {}'.format(matrix_path))

    with open(result_path, 'w') as f:
        f.write('=' * 60 + '\n')
        f.write('results ({})\n'.format(model_name))
        f.write('=' * 60 + '\n')
        f.write('checkpoint: {}\n'.format(checkpoint_path))
        f.write('OA: {:.5f}\n'.format(pixelAcc))
        f.write('mIoU: {:.5f}\n'.format(mean_iou))
        f.write('Macro F1: {:.5f}\n'.format(macro_f1))

    print('detailed results saved: {}'.format(result_path))

    return {
        'OA': pixelAcc,
        'mIoU': mean_iou,
        'Macro_F1': macro_f1,
        'model': model_name,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate EAGG-UNet')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint path (default inferred from config)')
    parser.add_argument('--config', type=str, default='configs/eval.json',
                        help='path to the evaluation config json')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = json.load(f)

    evaluate(config, checkpoint_path=args.checkpoint)


if __name__ == '__main__':
    main()
