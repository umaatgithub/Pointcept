"""
Preprocessing Script for Touchstone3D
Parsing normal vectors has a large consumption of memory. Please reduce max_workers if memory is limited.

Author: Umamaheswaran Raman Kumar & Abdur R. Fayjie, 2023
Please cite our work if the code is helpful to you.
"""

import os
import argparse
import glob
import torch
import numpy as np
import multiprocessing as mp

try:
    import open3d
except ImportError:
    import warnings
    warnings.warn(
        'Please install open3d for parsing normal')

try:
    import trimesh
except ImportError:
    import warnings
    warnings.warn(
        'Please install trimesh for parsing normal')

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def parse_room(room, dataset_root, output_root, classes, color2class, class2label):
    [house, room] = room
    print(house, room)
    source_path = os.path.join(dataset_root, house, room)
    f_split = room.split('.')
    f_split = [f for f in f_split if f not in ('obj', 'groundtruth', 'pcd')]
    room = 'Floor'+f_split[0] + '_'+''.join(f_split[1:])
    save_path = os.path.join(output_root, house, room) + '.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pcd = open3d.io.read_point_cloud(source_path)
    room_coords = np.asarray(pcd.points) / 100.0
    colors_gt = np.asarray(pcd.colors) * 255
    colors_gt = colors_gt.astype('int')
    room_semantic_gt = np.ones((colors_gt.shape[0],1), dtype=int) * -1
    
    for i in range(0,room_semantic_gt.shape[0]):
        (r,g,b) = (colors_gt[i][0],colors_gt[i][1],colors_gt[i][0])
        if (r,g,b) in color2class.keys():
            room_semantic_gt[i] = class2label[color2class[(r,g,b)]]
        else:
            room_semantic_gt[i] = -1

    save_dict = dict(coord=room_coords, semantic_gt=room_semantic_gt)
    torch.save(save_dict, save_path)


def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', required=True, help='Path to Touchstone dataset')
    parser.add_argument('--output_root', required=True, help='Output path where house folders will be located')
    args = parser.parse_args()
    
    classes = list([x.rstrip().split()[0]
                   for x in open(os.path.join(args.dataset_root, "raw", "meta", "touchstone3d.txt"))])
    color2class = {tuple(map(int, x.rstrip().split()[1:4])): x.rstrip().split()[0]
                   for x in open(os.path.join(args.dataset_root, "raw", "meta", "touchstone3d.txt"))}
    class2label = {cls: i for i, cls in enumerate(classes)}
    data_path = args.dataset_root+"/raw/data"
    
    room_list = []
    # Load room information
    for i in range(1, 33):
        for file_path in glob.glob(data_path+"/House{}".format(i)+"/*.pcd"):
            room_name = os.path.split(file_path)[1]
            room_list.append(["House{}".format(i), room_name])

    # Preprocess data
    print('Processing scenes...')
    # pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
    pool = ProcessPoolExecutor(max_workers=8)
    _ = list(pool.map(
        parse_room, room_list, 
        repeat(data_path), repeat(args.output_root), repeat(classes), repeat(color2class), repeat(class2label)
    ))


if __name__ == '__main__':
    main_process()
