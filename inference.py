import pathlib
import sys
import os
from typing import Any

import numpy as np
from common.utils import Progbar, read_annotations, write_csv_file
import torch.backends.cudnn as cudnn
from models.mvssnet import get_mvss
from models.resfcn import ResFCN
import torch.utils.data
from common.tools import inference_single
import cv2
from apex import amp
import argparse


def get_opt():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument("--model_path", type=str, default="ckpt/mvssnet.pth",
                        help="Path to the pretrained model")
    parser.add_argument("--test_file", type=str,
                        help="Path to the image list. It can be either "
                             "a TXT file or a CSV file with at least `image`, " 
                             "`mask` and `prediction` columns.")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--resize", type=int, default=512)
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = get_opt()
    print("in the head of inference:", opt)
    cudnn.benchmark = True

    # read test data
    test_file = opt.test_file
    dataset_name = os.path.basename(test_file).split('.')[0]
    model_type = os.path.basename(opt.model_path).split('.')[0]
    if not os.path.exists(test_file):
        print("%s not exists,quit" % test_file)
        sys.exit()
    test_data = read_annotations(test_file)
    new_size = opt.resize

    # load model
    model_path = opt.model_path
    if "mvssnet" in model_path:
        model = get_mvss(backbone='resnet50',
                         pretrained_base=True,
                         nclass=1,
                         sobel=True,
                         constrain=True,
                         n_input=3,
                         )
    elif "fcn" in model_path:
        model = ResFCN()
    else:
        print("model not found ", model_path)
        sys.exit()

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint["model_dict"], strict=False)
        model.eval()
        print("load %s finish" % (os.path.basename(model_path)))
    else:
        print("%s not exist" % model_path)
        sys.exit()
    model.cuda()
    amp.register_float_function(torch, 'sigmoid')
    model = amp.initialize(models=model, opt_level='O1', loss_scale='dynamic')
    model.eval()

    save_path = os.path.join(opt.save_dir, dataset_name, model_type)
    print("predicted maps will be saved in :%s" % save_path)
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        progbar = Progbar(len(test_data), stateful_metrics=['path'])
        pd_img_lab = []
        lab_all = []
        scores = []

        authentic_detections: list[dict[str, Any]] = []
        manipulated_detections: list[dict[str, Any]] = []

        for ix, (img_path, _, detection_label) in enumerate(test_data):
            img = cv2.imread(img_path)
            ori_size = img.shape
            img = cv2.resize(img, (new_size, new_size))
            seg, predicted_detection = inference_single(img=img, model=model, th=0)
            if detection_label == 1:
                # Save predicted results for the manipulated samples in the dataset.
                manipulated_detections.append({
                    "image": pathlib.Path(img_path).name,
                    "mvssnet_detection": predicted_detection
                })
                save_dir_path = pathlib.Path(save_path) / 'pred' / 'manipulated_masks'

            else:
                # Save predicted results for the authentic samples in the dataset.
                authentic_detections.append({
                    "image": pathlib.Path(img_path).name,
                    "mvssnet_detection": predicted_detection
                })
                save_dir_path = pathlib.Path(save_path) / 'pred' / 'authentic_masks'
            save_dir_path.mkdir(exist_ok=True, parents=True)
            save_seg_path = save_dir_path / f'{pathlib.Path(img_path).stem}.png'
            seg = cv2.resize(seg, (ori_size[1], ori_size[0]))
            cv2.imwrite(str(save_seg_path), seg.astype(np.uint8))
            progbar.add(1, values=[('path', save_seg_path), ])

        # Save detection CSVs.
        if len(authentic_detections) > 0:
            write_csv_file(authentic_detections,
                           pathlib.Path(save_path)/"pred"/"authentic_masks"/"detection_results.csv")
        if len(manipulated_detections) > 0:
            write_csv_file(manipulated_detections,
                           pathlib.Path(save_path)/"pred"/"manipulated_masks"/"detection_results.csv")
