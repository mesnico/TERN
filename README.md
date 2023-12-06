# Transformer Encoder Reasoning Network

## Updates

- :fire: 09/2022: The extension to this work (**ALADIN: Distilling Fine-grained Alignment Scores for Efficient Image-Text Matching and Retrieval**) has been published in proceedings of CBMI 2022. Check out [code](https://github.com/mesnico/ALADIN) and [paper](https://arxiv.org/abs/2207.14757)!

## Introduction

Code for the cross-modal visual-linguistic retrieval method from "Transformer Reasoning Network for Image-Text Matching and Retrieval", accepted to ICPR 2020 [[Pre-print PDF](https://arxiv.org/pdf/2004.09144.pdf)].

This repo is built on top of [VSE++](https://github.com/fartashf/vsepp).
<p align="center">
  <img src="images/architecture.png">
</p>


## Setup

1. Clone the repo and move into it:
```
git clone https://github.com/mesnico/TERN
cd TERN
```

2. Setup python environment using conda:
```
conda env create --file environment.yml
conda activate tern
export PYTHONPATH=.
```

## Get the data

Data and pretrained models be downloaded from this [OneDrive link](https://cnrsc-my.sharepoint.com/:f:/g/personal/nicola_messina_cnr_it/EnsuSFo-rG5Pmf2FhQDPe7EBCHrNtR1ujSIOEcgaj5Xrwg?e=Ger6Sl) (see the steps below to understand which files you need):


1. Download and extract the data folder, containing COCO annotations, the splits by Karpathy et al. and ROUGEL - SPICE precomputed relevances:

```
tar -xvf data.tgz
```

2. Download the bottom-up features. We rearranged the ones provided by [Anderson et al.](https://github.com/peteanderson80/bottom-up-attention) in multiple .npy files, one for every image in the COCO dataset. This is beneficial during the dataloading phase.
The following command extracts them under `data/coco/`. If you prefer another location, be sure to adjust the configuration file accordingly.
```
tar -xvf features_36_coco.tgz -C data/coco
```

## Evaluate
Download our pre-trained TERN model from the aforementioned link and extract it:
```
tar -xvf TERN_model_best_ndcg.pth.tgz
```

Then, issue the following commands for evaluating the model on the 1k (5fold cross-validation) or 5k test sets.
```
python3 test.py model_best_ndcg.pth --config configs/tern.yaml --size 1k
python3 test.py model_best_ndcg.pth --config configs/tern.yaml --size 5k
```

## Train
In order to train the model using the basic TERN configuration, issue the following command:
```
python3 train.py --config configs/tern.yaml --logger_name runs/tern
```
`runs/tern` is where the output files (tensorboard logs, checkpoints) will be stored during this training session.

## Reference
If you found this code useful, please cite the following paper:

    @inproceedings{messina2021transformer,
      title={Transformer reasoning network for image-text matching and retrieval},
      author={Messina, Nicola and Falchi, Fabrizio and Esuli, Andrea and Amato, Giuseppe},
      booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
      pages={5222--5229},
      year={2021},
      organization={IEEE}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
