import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.utils import l2norm
from transformers import BertTokenizer, BertModel, BertConfig


def EncoderText(config):
    num_layers = config['text-model']['layers']
    order_embeddings = config['training']['measure'] == 'order'
    if config['text-model']['name'] == 'bert':
        print('Using BERT text encoder')
        model = EncoderTextBERT(config, order_embeddings=order_embeddings, post_transformer_layers=num_layers)
    else:
        model = None
    return model


class EncoderTextBERT(nn.Module):
    def __init__(self, config, order_embeddings=False, mean=True, post_transformer_layers=0):
        super().__init__()
        self.preextracted = config['text-model']['pre-extracted']
        bert_config = BertConfig.from_pretrained(config['text-model']['pretrain'],
                                                 output_hidden_states=True,
                                                 num_hidden_layers=config['text-model']['extraction-hidden-layer'])
        bert_model = BertModel.from_pretrained(config['text-model']['pretrain'], config=bert_config)
        self.order_embeddings = order_embeddings
        self.vocab_size = bert_model.config.vocab_size
        self.hidden_layer = config['text-model']['extraction-hidden-layer']
        if not self.preextracted:
            self.tokenizer = BertTokenizer.from_pretrained(config['text-model']['pretrain'])
            self.bert_model = bert_model
            self.word_embeddings = self.bert_model.get_input_embeddings()
        if post_transformer_layers > 0:
            transformer_layer = nn.TransformerEncoderLayer(d_model=config['text-model']['word-dim'], nhead=4,
                                                           dim_feedforward=2048,
                                                           dropout=config['text-model']['dropout'], activation='relu')
            self.transformer_encoder = nn.TransformerEncoder(transformer_layer,
                                                             num_layers=post_transformer_layers)
        self.post_transformer_layers = post_transformer_layers
        self.map = nn.Linear(config['text-model']['word-dim'], config['model']['embed-size'])
        self.mean = mean

    def forward(self, x, lengths):
        '''
        x: tensor of indexes (LongTensor) obtained with tokenizer.encode() of size B x ?
        lengths: tensor of lengths (LongTensor) of size B
        '''
        if not self.preextracted or self.post_transformer_layers > 0:
            max_len = max(lengths)
            attention_mask = torch.ones(x.shape[0], max_len)
            for e, l in zip(attention_mask, lengths):
                e[l:] = 0
            attention_mask = attention_mask.to(x.device)

        if self.preextracted:
            outputs = x
        else:
            outputs = self.bert_model(x, attention_mask=attention_mask)
            outputs = outputs[2][-1]

        if self.post_transformer_layers > 0:
            outputs = outputs.permute(1, 0, 2)
            outputs = self.transformer_encoder(outputs, src_key_padding_mask=(attention_mask - 1).bool())
            outputs = outputs.permute(1, 0, 2)
        if self.mean:
            x = outputs.mean(dim=1)
        else:
            x = outputs[:, 0, :]     # from the last layer take only the first word

        out = self.map(x)

        # normalization in the joint embedding space
        # out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.order_embeddings:
            out = torch.abs(out)
        return out, outputs

    def get_finetuning_params(self):
        return list(self.bert_model.parameters())
