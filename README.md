# Automatic Fish Segmentation (AFS)

[![DOI](https://zenodo.org/badge/113559427.svg)](https://zenodo.org/badge/latestdoi/113559427)

AFS is a Python 2.7 application for automated segmentation of medaka fish.

## Features
* Multi-scale approach
* Alignemt of a fish by three landmarks in eyes and tail
* Smart separation data to a head and tail part
* Organ localization by elastic deformation provided by ANTs
* Segmentation of organs in head part [eyes, brain]
* Segmentation of organs in tail part [spinal cord, heart, liver] [muscles, kidney, intestine, spleen are in process]

## Required software
* Advanced Normalization Tools (ANTs) (https://github.com/ANTsX/ANTs)

## Installation
Clone the current version from GitHub by opening a terminal and type in the following:
```bash
$ git clone https://github.com/AFSRepo/AFS-Segmentation.git
$ cd AFS-Segmentation
$ export PATH=~/antsbin/bin:$PATH
$ export ANTSPATH=~/antsbin/bin
```
## Run
Setup variable INPUT_DIR of main.py to the directory containing folders with datasets in *.raw format named as fishXXX_##bit_WIDTHxHEIGHTxDEPTH.raw. The variable OUTPUT_DIR should point to the directory where you are going to store produced results.
When everything is setup, you can run the process with:
```bash
$ cd AFS-Segmentation
$ python main.py
```


