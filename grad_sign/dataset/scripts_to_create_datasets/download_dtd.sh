#!/bin/bash -l
#SBATCH --job-name=Task
#SBATCH --partition=all_serial
#SBATCH -wailb-login-03
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --account=debiasing

cd /work/debiasing/datasets/
mkdir dtd && cd dtd
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xvzf dtd-r1.0.1.tar.gz
rm -rf dtd-r1.0.1.tar.gz
mv dtd/images images
mv dtd/imdb/ imdb
mv dtd/labels labels
cat labels/train1.txt labels/val1.txt > labels/train.txt
cat labels/test1.txt > labels/test.txt
