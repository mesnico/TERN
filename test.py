import argparse

import evaluation
import yaml
import torch

def main(opt, current_config):
    model_checkpoint = opt.checkpoint

    checkpoint = torch.load(model_checkpoint)
    print('Checkpoint loaded from {}'.format(model_checkpoint))
    loaded_config = checkpoint['config']

    if opt.size == "1k":
        fold5 = True
    elif opt.size == "5k":
        fold5 = False
    else:
        raise ValueError('Test split size not recognized!')

    # Override some mandatory things in the configuration (paths)
    loaded_config['dataset']['images-path'] = current_config['dataset']['images-path']
    loaded_config['dataset']['data'] = current_config['dataset']['data']
    loaded_config['image-model']['pre-extracted-features-root'] = current_config['image-model']['pre-extracted-features-root']

    evaluation.evalrank(loaded_config, checkpoint, split="test", fold5=fold5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help="Checkpoint to load")
    parser.add_argument('--size', type=str, choices=['1k', '5k'], default='1k')
    parser.add_argument('--config', type=str, help="Which configuration to use. See into 'config' folder")

    opt = parser.parse_args()
    with open(opt.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)
    main(opt, config)
