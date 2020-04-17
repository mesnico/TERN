# This script extract features and put them in shelve format

import os
import torch
import tqdm
import argparse
import yaml
import re
import itertools
import pickle
import numpy as np
from torch.utils.data import DataLoader
# from graphrcnn.extract_features import extract_visual_features
# from torchvision.datasets.coco import CocoCaptions
# from datasets import CocoCaptionsOnly
from torchvision import transforms
from torchvision.models import resnet18, resnet50, resnet101, resnet152, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from transformers import BertTokenizer, BertModel
# from datasets import TextCollator
import shelve
import data
from models.text import EncoderTextBERT


class TextCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched captions.
    This should be passed to the DataLoader
    """

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # images = transposed_batch[0]
        captions = transposed_batch[1]
        return captions


class FeatureExtractor(object):
    def __init__(self, config, split, bs=1, collate_fn=torch.utils.data.dataloader.default_collate):
        self.config = config
        self.split = split
        self.output_feat_fld = os.path.join(config['dataset']['data'], '{}_precomp'.format(config['dataset']['name']))
        if not os.path.exists(self.output_feat_fld):
            os.makedirs(self.output_feat_fld)

    def extract(self):
        """
        Extracts features and dump them on a db file.
        For text extractors: each file record is a dictionary with keys:
        'image_id' (int) and 'features' (np.array K x dim)
        For image extractors: each file record is a dictionary with keys:
        'boxes' (np.array K x 4), 'scores' (np.array K x 1), 'features' (np.array K x dim)
        :return: void
        """
        raise NotImplementedError

    def get_db_file(self):
        """
        :return: the path to the db file for these features
        """
        raise NotImplementedError


class HuggingFaceTransformerExtractor(FeatureExtractor):
    def __init__(self, config, split, model_name='bert', pretrained='bert-base-uncased', finetuned=None):
        super(HuggingFaceTransformerExtractor, self).__init__(config, split, bs=5, collate_fn=TextCollator())
        self.pretrained = pretrained
        self.finetuned = finetuned
        self.model_name = model_name
        self.config = config

        roots, ids = data.get_paths(config)

        data_name = config['dataset']['name']
        transform = data.get_transform(data_name, 'val', config)
        collate_fn = data.Collate(config)
        self.loader = data.get_loader_single(data_name, split,
                                             roots[split]['img'],
                                             roots[split]['cap'],
                                             transform, ids=ids[split],
                                             batch_size=32, shuffle=False,
                                             num_workers=4, collate_fn=collate_fn)

    def get_db_file(self):
        finetuned_str = "" if not self.finetuned else '_finetuned'
        feat_dst_filename = os.path.join(self.output_feat_fld,
                                         '{}_{}_{}{}.db'.format(self.split, self.model_name, self.pretrained, finetuned_str))
        print('Hugging Face BERT features filename: {}'.format(feat_dst_filename))
        return feat_dst_filename

    def extract(self, device='cuda'):
        # Load pre-trained model tokenizer (vocabulary) and model itself
        if self.model_name == 'bert':
            self.config['text-model']['layers'] = 0
            self.config['text-model']['pre-extracted'] = False
            model = EncoderTextBERT(self.config)
        else:
            raise ValueError('{} model is not known'.format(self.model))

        if self.finetuned:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            checkpoint = torch.load(self.finetuned, map_location=device)['model']
            checkpoint = {k[k.find('.txt_enc.'):].replace('.txt_enc.', ''): v for k, v in checkpoint.items() if '.txt_enc.' in k}
            model.load_state_dict(checkpoint, strict=False)
            print('BERT model extracted from trained model at {}'.format(self.finetuned))

        model.to(device)
        model.eval()

        feat_dst_filename = self.get_db_file()
        prog_id = 0
        with shelve.open(feat_dst_filename, flag='n') as db:
            for images, captions, img_lengths, cap_lengths, boxes, ids in tqdm.tqdm(self.loader):
                captions = captions.cuda()

                with torch.no_grad():
                    _, feats = model(captions, cap_lengths)
                    # get the features from the last hidden state
                    feats = feats.cpu().numpy()
                    word_embs = model.word_embeddings(captions)
                    word_embs = word_embs.cpu().numpy()
                    for c, f, w, l, i in zip(captions.cpu().numpy(), feats, word_embs, cap_lengths, ids):
                        # dump_feats.append(f[:l])
                        dump_dict = {'image_id': i, 'captions': c, 'features': f[:l], 'wembeddings': w[:l]}
                        db[str(prog_id)] = dump_dict
                        prog_id += 1


def get_features_extractor(config, split, method=None, finetuned=None):
    if method == 'transformer-bert':
        config['text-model']['pre-extracted'] = False
        extractor = HuggingFaceTransformerExtractor(config, split, finetuned=finetuned)

    # elif method == 'graphrcnn':
    #     extractor = GraphRcnnFeatureExtractor(dataset_name, dataset_root, split,
    #                                           extractor_config['algorithm'])
    # elif method == 'resnet':
    #     extractor = ResnetFeatureExtractor(dataset_name, dataset_root, split,
    #                                        extractor_config['depth'], (extractor_config['output-h'],
    #                                                                    extractor_config['output-w']))
    # elif method == 'vgg':
    #     extractor = VGGFeatureExtractor(dataset_name, dataset_root, split,
    #                                     extractor_config['depth'])
    else:
        raise ValueError('Extraction method {} not known!'.format(args.method))
    return extractor


def main(args, config):
    extractor = get_features_extractor(config, args.split, args.method, args.finetuned)
    if os.path.isfile(extractor.get_db_file() + '.dat'):
        answ = input("Features {} for {} already existing. Overwrite? (y/n)".format(extractor.get_db_file(), extractor))
        if answ == 'y':
            print('Using extractor: {}'.format(extractor))
            extractor.extract()
        else:
            print('Skipping {}'.format(extractor))
    else:
        extractor.extract()

    print('DONE')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Extract captioning scores for use as relevance')
    arg_parser.add_argument('--config', type=str, help="Which configuration to use. See into 'config' folder")
    arg_parser.add_argument('--split', type=str, default="val", help="Dataset split to use")
    arg_parser.add_argument('--finetuned', type=str, default=None, help="Optional finetuning checkpoint")
    arg_parser.add_argument('method', type=str, help="Which kind of feature you want to extract")
    # arg_parser.add_argument('type', type=str, choices=['image','text'], help="Method type")

    args = arg_parser.parse_args()

    if args.finetuned is not None:
        config = torch.load(args.finetuned)['config']
        print('Configuration read from checkpoint')
    else:
        with open(args.config, 'r') as ymlfile:
            config = yaml.load(ymlfile)
    main(args, config)

