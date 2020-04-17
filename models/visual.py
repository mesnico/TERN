from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
from torchvision import models as models
from models.utils import PositionalEncodingImageBoxes, l2norm


def EncoderImage(config):

    # data_name, img_dim, embed_size, finetune=False,
    #         cnn_type='vgg19', use_abs=False, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """

    embed_size = config['model']['embed-size']
    order_embeddings = config['training']['measure'] == 'order'
    if config['image-model']['name'] == 'bottomup':
        transformer_layers = config['image-model']['transformer-layers']
        pos_encoding = config['image-model']['pos-encoding']
        visual_feat_dim = config['image-model']['feat-dim']
        dropout = config['image-model']['dropout']
        img_enc = TransformerPostProcessing(transformer_layers, visual_feat_dim, embed_size, n_head=4, aggr='mean', pos_encoding=pos_encoding, dropout=dropout, order_embeddings=order_embeddings)
    else:
        img_enc = None

    return img_enc


class TransformerPostProcessing(nn.Module):
    def __init__(self, num_transformer_layers, feat_dim, embed_size, n_head=4, aggr='mean', pos_encoding=None, dropout=0.1, order_embeddings=False):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=n_head,
                                                       dim_feedforward=2048,
                                                       dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer,
                                                         num_layers=num_transformer_layers)
        if pos_encoding is not None:
            self.pos_encoding_image = PositionalEncodingImageBoxes(feat_dim, pos_encoding)
        self.fc = nn.Linear(feat_dim, embed_size)
        self.aggr = aggr
        self.order_embeddings = order_embeddings
        if aggr == 'gated':
            self.gate_fn = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, 1)
            )
            self.node_fn = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim)
            )
        self.pos_encoding = pos_encoding

    def forward(self, visual_feats, visual_feats_len=None, boxes=None):
        """
        Takes an variable len batch of visual features and preprocess them through a transformer. Output a tensor
        with the same shape as visual_feats passed in input.
        :param visual_feats:
        :param visual_feats_len:
        :return: a tensor with the same shape as visual_feats passed in input.
        """
        # max_len = max(visual_feats_len)
        # bs = visual_feats.shape[1]
        # attention_mask = torch.zeros(bs, max_len).bool()
        # for e, l in zip(attention_mask, visual_feats_len):
        #     e[l:] = True
        # attention_mask = attention_mask.to(visual_feats.device)

        visual_feats = visual_feats.permute(1, 0, 2)
        if self.pos_encoding is not None:
            visual_feats = self.pos_encoding_image(visual_feats, boxes)

        if visual_feats_len is not None:
            bs = visual_feats.shape[1]
            # construct the attention mask
            max_len = max(visual_feats_len)
            mask = torch.zeros(bs, max_len).bool()
            for e, l in zip(mask, visual_feats_len):
                e[l:] = True
            mask = mask.to(visual_feats.device)
        else:
            mask = None

        visual_feats = self.transformer_encoder(visual_feats, src_key_padding_mask=mask)
        # visual_feats = visual_feats.permute(1, 0, 2)

        if self.aggr == 'mean':
            out = visual_feats.mean(dim=0)
        elif self.aggr == 'gated':
            out = visual_feats.permute(1, 0, 2)
            m = torch.sigmoid(self.gate_fn(out))   # B x S x 1
            v = self.node_fn(out)   # B x S x dim
            out = torch.bmm(m.permute(0, 2, 1), v)      # B x 1 x dim
            out = out.squeeze(1)    # B x dim
        else:
            out = visual_feats[0]

        out = self.fc(out)
        if self.order_embeddings:
            out = torch.abs(out)

        return out, visual_feats.permute(1, 0, 2)


def find_nhead(feat_dim, higher=8):
    # find the right n_head value (the highest value lower than 'higher')
    for i in reversed(range(higher + 1)):
        if feat_dim % i == 0:
            return i
    return 1