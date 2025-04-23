#!/usr/bin/env bash
# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

PETS_IMAGES_PATH=https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
PETS_ANNOTATIONS_PATH=https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz
UCI_DATASET_PATH=https://github.com/aleximmer/heteroscedastic-nn/archive/refs/heads/main.zip
CQR_DATASET_PATH=https://github.com/yromano/cqr/archive/refs/heads/master.zip
FACEBOOK_DATASET_PATH=https://archive.ics.uci.edu/static/public/363/facebook+comment+volume+dataset.zip
BLOG_DATASET_PATH=https://archive.ics.uci.edu/static/public/304/blogfeedback.zip

# Oxford Pets
mkdir data
mkdir data/oxford-iiit-pet
cd data/oxford-iiit-pet
wget $PETS_IMAGES_PATH
tar -xvzf images.tar.gz
wget $PETS_ANNOTATIONS_PATH
tar -xvzf annotations.tar.gz
rm -rf images.tar.gz annotations.tar.gz
cd ..
# UCI datasets
mkdir uci
cd uci
wget $UCI_DATASET_PATH
unzip main.zip
cp -r heteroscedastic-nn-main/data/* .
rm -rf heteroscedastic-nn-main main.zip crispr
# Renaming dataset folders
mv boston-housing/ boston
mv wine-quality-red/ wine
mv power-plant/ power
mv naval-propulsion-plant/ naval
cd ..
# CQR datasets
mkdir cqr
cd cqr
wget $CQR_DATASET_PATH
unzip master.zip
cp -r cqr-master/datasets/* .
rm -rf cqr-master master.zip Concrete_Data.csv datasets.py
# Facebook dataset
wget $FACEBOOK_DATASET_PATH
unzip facebook+comment+volume+dataset.zip
cp Dataset/Training/Features_Variant_1.csv facebook/
cp Dataset/Training/Features_Variant_2.csv facebook/
rm -rf facebook+comment+volume+dataset.zip __MACOSX Dataset README.md
# Blog dataset
mkdir blog
cd blog
wget $BLOG_DATASET_PATH
unzip blogfeedback.zip
rm -f blogfeedback.zip
cp blogData_train.csv ../
cd ..
rm -rf blog
