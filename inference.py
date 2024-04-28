import pathlib
import sys
import os
import timeit
from typing import Any, Optional

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

from common import csv_utils


def get_opt():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument("--model_path", type=str, default="ckpt/mvssnetplus_casia.pt",
                        help="Path to the pretrained model")
    parser.add_argument("--test_file", type=str,
                        help="Path to the image list. It can be either "
                             "a TXT file or a CSV file with at least `image`, " 
                             "`mask` and `prediction` columns.")
    parser.add_argument("--save_dir", type=pathlib.Path, default="")
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("-r", "--csv_root_dir", type=pathlib.Path, default=None)
    parser.add_argument("--update_input_csv", action="store_true")
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = get_opt()
    print("in the head of inference:", opt)
    cudnn.benchmark = True

    csv_root_dir: Optional[pathlib.Path] = opt.csv_root_dir
    update_input_csv: bool = opt.update_input_csv

    # read test data
    test_file = opt.test_file
    dataset_name = os.path.basename(test_file).split('.')[0]
    model_type = os.path.basename(opt.model_path).split('.')[0]
    if not os.path.exists(test_file):
        print("%s not exists,quit" % test_file)
        sys.exit()
    test_data = read_annotations(test_file, root_path=csv_root_dir)
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

    save_path: pathlib.Path = opt.save_dir
    print("Predicted maps will be saved in :%s" % save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        progbar = Progbar(len(test_data), stateful_metrics=['path'])
        pd_img_lab = []
        lab_all = []
        scores = []

        authentic_detections: list[dict[str, Any]] = []
        manipulated_detections: list[dict[str, Any]] = []
        predicted_masks: dict[pathlib.Path, pathlib] = {}
        predicted_detection_scores: dict[pathlib.Path, float] = {}

        torch.cuda.synchronize()
        start_time: float = timeit.default_timer()

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
                save_dir_path = save_path / 'manipulated_masks'
            else:
                # Save predicted results for the authentic samples in the dataset.
                authentic_detections.append({
                    "image": pathlib.Path(img_path).name,
                    "mvssnet_detection": predicted_detection
                })
                save_dir_path = save_path / 'authentic_masks'
            save_dir_path.mkdir(exist_ok=True, parents=True)
            save_seg_path = save_dir_path / f'{pathlib.Path(img_path).stem}.png'
            seg = cv2.resize(seg, (ori_size[1], ori_size[0]))
            cv2.imwrite(str(save_seg_path), seg.astype(np.uint8))
            predicted_masks[pathlib.Path(img_path)] = save_seg_path
            predicted_detection_scores[pathlib.Path(img_path)] = predicted_detection
            progbar.add(1, values=[('path', save_seg_path), ])

        torch.cuda.synchronize()
        stop_time: float = timeit.default_timer()
        elapsed_time: float = stop_time - start_time
        print(f"Total time: {elapsed_time} secs")
        print(f"Time per image: {elapsed_time / len(test_data)}")

        # Save detection CSVs.
        if len(authentic_detections) > 0:
            write_csv_file(authentic_detections,
                           pathlib.Path(save_path)/"authentic_masks"/"detection_results.csv")
        if len(manipulated_detections) > 0:
            write_csv_file(manipulated_detections,
                           pathlib.Path(save_path)/"manipulated_masks"/"detection_results.csv")

        if update_input_csv:
            output_data: csv_utils.AlgorithmOutputData = csv_utils.AlgorithmOutputData(
                tampered=predicted_masks,
                untampered=None,
                response_times=None,
                image_level_predictions=predicted_detection_scores
            )
            output_data.save_csv(
                pathlib.Path(opt.test_file),
                root_path=csv_root_dir,
                output_column="mvssnetplus"
            )
