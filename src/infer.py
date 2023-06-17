import argparse
import os

import torch
from path import Path

from WordDetectorNN.src.dataloader import DataLoaderImgFile
from WordDetectorNN.src.eval import evaluate
from WordDetectorNN.src.net import WordDetectorNet
from WordDetectorNN.src.visualization import visualize_and_plot, crop_and_save

def main(path_to_test_folder):
    imgs = []
    AABBs = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = 'cpu'
    net = WordDetectorNet()
    net.load_state_dict(torch.load('../WordDetectorNN/model/weights', map_location=args.device))
    net.eval()
    net.to(args.device)

    loader = DataLoaderImgFile(Path(path_to_test_folder), net.input_size, args.device)
    res = evaluate(net, loader, max_aabbs=1000)
    temp_folder_path = "../temp/"
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)
    counter = 1
    for i, (img, aabbs) in enumerate(zip(res.batch_imgs, res.batch_aabbs)):
        f = loader.get_scale_factor(i)
        aabbs = [aabb.scale(1 / f, 1 / f) for aabb in aabbs]
        img = loader.get_original_img(i)
        counter = crop_and_save(counter, img, aabbs, path_to_save=temp_folder_path)
        imgs.append(img)
        AABBs.append(aabbs)
    return (temp_folder_path, imgs, AABBs)


if __name__ == '__main__':
    main("../test")
