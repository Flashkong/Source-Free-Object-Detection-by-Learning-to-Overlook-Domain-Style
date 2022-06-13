# coding:utf-8
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.cartoon_voc import cartoon_voc
from datasets.comic_voc_ import comic_voc
from datasets.watercolor_voc import watercolor_voc

__sets = {}
from datasets.pascal_voc import pascal_voc

from datasets.imagenet import imagenet
from datasets.vg import vg

from datasets.cityscape import cityscape
from datasets.KITTI import KITTI
from datasets.SIM import SIM
from datasets.KITTI_cityscape import KITTI_cityscape
from datasets.SIM_cityscape import SIM_cityscape
from datasets.clipart import clipart
from datasets.cartoon import cartoon
from datasets.watercolor import watercolor
from datasets.comic import comic
from datasets.bdd_cityscape import bdd_cityscape
from datasets.bdd import bdd
import os
from model.utils.config import cfg

#  clipart for voc2clipart
for year in ['2007']:
  for split in ['train_t', 'test_t', 'traintest1k']:
    name = 'clipart_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: clipart(split, year, os.path.join(cfg.DATA_DIR, 'clipart/')))

# cartoon for voc2cartoon
for year in ['2007']:
  for split in ['train_cartoon', 'test_cartoon']:
    name = 'cartoon_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: cartoon(split, year, os.path.join(cfg.DATA_DIR, 'cartoon/')))

# watercolor for voc2wc
for year in ['2007']:
  for split in ['train_wc', 'test_wc', 'train_wc1k', 'test_wc1k']:
    name = 'wc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: watercolor(split, year, os.path.join(cfg.DATA_DIR, 'watercolor/')))

# comic for voc2comic
for year in ['2007']:
  for split in ['train_comic', 'test_comic']:
    name = 'comic_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: comic(split, year, os.path.join(cfg.DATA_DIR, 'comic/')))

# cityscape and foggy cityscape
for year in ['2007', '2012']:
  for split in ['train_s', 'train_t', 'train_all', 'test_s', 'test_s500', 'test_t','test_all']:
    name = 'cityscape_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: cityscape(split, year, os.path.join(cfg.DATA_DIR, 'cityscape/')))

# bdd for city2bdd
for year in ['2007']:
  for split in ['train', 'val','valall']:
    name = 'bdd_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: bdd(split, os.path.join(cfg.DATA_DIR, 'bdd100k_voc/')))

# cityscape for city2bdd
for year in ['2007']:
  for split in ['train_s', 'train_t', 'train_all', 'test_s', 'test_s500', 'test_t','test_all']:
    name = 'bdd_cityscape_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: bdd_cityscape(split, year, os.path.join(cfg.DATA_DIR, 'cityscape/')))


for year in ['2007']:
  for split in ['trainall', 'train500']:
    name = 'KITTI_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: KITTI(split, year, os.path.join(cfg.DATA_DIR, 'KITTI/')))
  for year in ['2007']:
      for split in ['train_s', 'test_s', 'train_t','test_t', 'test_s500']:
          name = 'KITTI_cityscape_{}_{}'.format(year, split)
          __sets[name] = (lambda split=split, year=year: KITTI_cityscape(split, year, os.path.join(cfg.DATA_DIR, 'cityscape/')))

for year in ['2012']:
  for split in ['train_s']:
    name = 'SIM_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: SIM(split, year, os.path.join(cfg.DATA_DIR, 'SIM/')))

# cityscape for SIM->cityscape
for year in ['2007']:
  for split in ['train_s', 'train_t', 'test_t', 'test_s500']:
    name = 'SIM_cityscape_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: SIM_cityscape(split, year, os.path.join(cfg.DATA_DIR, 'cityscape/')))

# pascal voc for voc2clipart
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, os.path.join(cfg.DATA_DIR, 'VOCdevkit/')))

# pascal voc for voc2cartoon
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'cartoon_voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: cartoon_voc(split, year, os.path.join(cfg.DATA_DIR, 'VOCdevkit/')))

# pascal voc for voc2wc
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'watercolor_voc_{}_{}'.format(year, split)
    __sets[name] = (
      lambda split=split, year=year: watercolor_voc(split, year, os.path.join(cfg.DATA_DIR, 'VOCdevkit/')))

# pascal voc for voc2comic
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'comic_voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: comic_voc(split, year, os.path.join(cfg.DATA_DIR, 'VOCdevkit/')))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
    for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
        name = 'vg_{}_{}'.format(version,split)
        __sets[name] = (lambda split=split, version=version: vg(version, split))
        
# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
