import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init

from models.loss import ContrastiveLoss
from models.text import EncoderText
from models.visual import EncoderImage
from .utils import l2norm, Aggregator


class JointTextImageTransformerEncoder(nn.Module):
    """
    This is a bert caption encoder - transformer image encoder (using bottomup features).
    If process the encoder outputs through a transformer, like VilBERT and outputs two different graph embeddings
    """
    def __init__(self, config):
        super().__init__()
        self.txt_enc = EncoderText(config)

        visual_feat_dim = config['image-model']['feat-dim']
        caption_feat_dim = config['text-model']['word-dim']
        dropout = config['model']['dropout']
        layers = config['model']['layers']
        embed_size = config['model']['embed-size']
        self.order_embeddings = config['training']['measure'] == 'order'
        self.img_enc = EncoderImage(config)

        self.img_proj = nn.Linear(visual_feat_dim, embed_size)
        self.cap_proj = nn.Linear(caption_feat_dim, embed_size)
        self.embed_size = embed_size
        self.shared_transformer = config['model']['shared-transformer']

        transformer_layer_1 = nn.TransformerEncoderLayer(d_model=embed_size, nhead=4,
                                                       dim_feedforward=2048,
                                                       dropout=dropout, activation='relu')
        self.transformer_encoder_1 = nn.TransformerEncoder(transformer_layer_1,
                                                           num_layers=layers)
        if not self.shared_transformer:
            transformer_layer_2 = nn.TransformerEncoderLayer(d_model=embed_size, nhead=4,
                                                             dim_feedforward=2048,
                                                             dropout=dropout, activation='relu')
            self.transformer_encoder_2 = nn.TransformerEncoder(transformer_layer_2,
                                                               num_layers=layers)
        self.text_aggregation = Aggregator(embed_size, aggregation_type=config['model']['text-aggregation'])
        self.image_aggregation = Aggregator(embed_size, aggregation_type=config['model']['image-aggregation'])
        self.text_aggregation_type = config['model']['text-aggregation']
        self.img_aggregation_type = config['model']['image-aggregation']

    def forward(self, features, captions, feat_len, cap_len, boxes):
        # process captions by using bert
        full_cap_emb_aggr, c_emb = self.txt_enc(captions, cap_len)     # B x S x cap_dim

        # process image regions using a two-layer transformer
        full_img_emb_aggr, i_emb = self.img_enc(features, feat_len, boxes)     # B x S x vis_dim
        # i_emb = i_emb.permute(1, 0, 2)                             # B x S x vis_dim

        bs = features.shape[0]

        # if False:
        #     # concatenate the embeddings together
        #     max_summed_lengths = max([x + y for x, y in zip(feat_len, cap_len)])
        #     i_c_emb = torch.zeros(bs, max_summed_lengths, self.embed_size)
        #     i_c_emb = i_c_emb.to(features.device)
        #     mask = torch.zeros(bs, max_summed_lengths).bool()
        #     mask = mask.to(features.device)
        #     for i_c, m, i, c, i_len, c_len in zip(i_c_emb, mask, i_emb, c_emb, feat_len, cap_len):
        #         i_c[:c_len] = c[:c_len]
        #         i_c[c_len:c_len + i_len] = i[:i_len]
        #         m[c_len + i_len:] = True
        #
        #     i_c_emb = i_c_emb.permute(1, 0, 2)      # S_vis + S_txt x B x dim
        #     out = self.transformer_encoder(i_c_emb, src_key_padding_mask=mask)      # S_vis + S_txt x B x dim
        #
        #     full_cap_emb = out[0, :, :]
        #     I = torch.LongTensor(cap_len).view(1, -1, 1)
        #     I = I.expand(1, bs, self.embed_size).to(features.device)
        #     full_img_emb = torch.gather(out, dim=0, index=I).squeeze(0)
        # else:

        # forward the captions
        if self.text_aggregation_type is not None:
            c_emb = self.cap_proj(c_emb)

            mask = torch.zeros(bs, max(cap_len)).bool()
            mask = mask.to(features.device)
            for m, c_len in zip(mask, cap_len):
                m[c_len:] = True
            full_cap_emb = self.transformer_encoder_1(c_emb.permute(1, 0, 2), src_key_padding_mask=mask)  # S_txt x B x dim
            full_cap_emb_aggr = self.text_aggregation(full_cap_emb, cap_len, mask)
        # else use the embedding output by the txt model
        else:
            full_cap_emb = None

        # forward the regions
        if self.img_aggregation_type is not None:
            i_emb = self.img_proj(i_emb)

            mask = torch.zeros(bs, max(feat_len)).bool()
            mask = mask.to(features.device)
            for m, v_len in zip(mask, feat_len):
                m[v_len:] = True
            if self.shared_transformer:
                full_img_emb = self.transformer_encoder_1(i_emb.permute(1, 0, 2), src_key_padding_mask=mask)  # S_txt x B x dim
            else:
                full_img_emb = self.transformer_encoder_2(i_emb.permute(1, 0, 2), src_key_padding_mask=mask)  # S_txt x B x dim
            full_img_emb_aggr = self.image_aggregation(full_img_emb, feat_len, mask)
        else:
            full_img_emb = None

        full_cap_emb_aggr = l2norm(full_cap_emb_aggr)
        full_img_emb_aggr = l2norm(full_img_emb_aggr)
        if self.order_embeddings:
            full_cap_emb_aggr = torch.abs(full_cap_emb_aggr)
            full_img_emb_aggr = torch.abs(full_img_emb_aggr)
        return full_img_emb_aggr, full_cap_emb_aggr, full_img_emb, full_cap_emb


class TERN(torch.nn.Module):

    def __init__(self, config):
        # tutorials/09 - Image Captioning
        # Build Models
        super().__init__()
        self.img_txt_enc = JointTextImageTransformerEncoder(config)
        if torch.cuda.is_available():
            self.img_txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=config['training']['margin'],
                                         measure=config['training']['measure'],
                                         max_violation=config['training']['max-violation'])

        self.Eiters = 0
        self.config = config

    # def state_dict(self):
    #     state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
    #     return state_dict
    #
    # def load_state_dict(self, state_dict):
    #     self.img_enc.load_state_dict(state_dict[0])
    #     self.txt_enc.load_state_dict(state_dict[1])
    #
    # def train_start(self):
    #     """switch to train mode
    #     """
    #     self.img_enc.train()
    #     self.txt_enc.train()
    #
    # def val_start(self):
    #     """switch to evaluate mode
    #     """
    #     self.img_enc.eval()
    #     self.txt_enc.eval()

    def forward_emb(self, images, captions, img_len, cap_len, boxes):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            boxes = boxes.cuda()

        # Forward
        img_emb_aggr, cap_emb_aggr, img_feats, cap_feats = self.img_txt_enc(images, captions, img_len, cap_len, boxes)
        return img_emb_aggr, cap_emb_aggr, img_feats, cap_feats

    def get_parameters(self):
        lr_multiplier = 1.0 if self.config['text-model']['fine-tune'] else 0.0

        ret = []
        params = list(self.img_txt_enc.img_enc.parameters())
        params += list(self.img_txt_enc.img_proj.parameters())
        params += list(self.img_txt_enc.cap_proj.parameters())
        params += list(self.img_txt_enc.transformer_encoder_1.parameters())

        params += list(self.img_txt_enc.image_aggregation.parameters())
        params += list(self.img_txt_enc.text_aggregation.parameters())

        if not self.config['model']['shared-transformer']:
            params += list(self.img_txt_enc.transformer_encoder_2.parameters())

        ret.append(params)

        ret.append(list(self.img_txt_enc.txt_enc.parameters()))

        return ret, lr_multiplier

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        bs = img_emb.shape[0]

        loss = self.criterion(img_emb, cap_emb)

        self.logger.update('Le', loss.item(), img_emb.size(0))
        return loss

    def forward(self, images, targets, img_lengths, cap_lengths, boxes=None, ids=None, *args):
        """One training step given images and captions.
        """
        # assert self.training()
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)

        if type(targets) == tuple or type(targets) == list:
            captions, features, wembeddings = targets
            # captions = features  # Very weird, I know
            text = features
        else:
            text = targets
            captions = targets
            wembeddings = self.img_txt_enc.txt_enc.word_embeddings(captions.cuda() if torch.cuda.is_available() else captions)

        # compute the embeddings
        img_emb_aggr, cap_emb_aggr, img_feats, cap_feats = self.forward_emb(images, text, img_lengths, cap_lengths, boxes)
        # NOTE: img_feats and cap_feats are S x B x dim

        retrieval_loss = self.forward_loss(img_emb_aggr, cap_emb_aggr)

        loss_dict = {'retrieval_loss': retrieval_loss}
        return loss_dict
