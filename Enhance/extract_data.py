from shutil import copy
from tqdm import tqdm
import argparse
import os
parser = argparse.ArgumentParser()

parser.add_argument('--file_path', type=str, required=True,
                    help='The path to the train.txt')
parser.add_argument('--images_folder', type=str, required=True,
                    help='The path to the images folder')
parser.add_argument('--scenario_name', type=str, required=True,choices=['voc2clipart', 'voc2wc', 'city2foggy' ,'KC'],help='The name of the scenario')
parser.add_argument('--image_suffix', type=str, default='jpg',help='image suffix')

args = parser.parse_args()

file_path=args.file_path
path=args.images_folder
dataset=args.scenario_name

with open(file_path,'r') as f:
    content=f.readlines()

dir=os.path.join('data',dataset)
if not os.path.exists(dir):
    os.makedirs(dir)

for i in tqdm(range(0,len(content))):
    file=os.path.join(path,content[i].split('\n')[0]+'.'+args.image_suffix)
    copy(file,dir+'/'+content[i].split('\n')[0]+'.'+args.image_suffix)