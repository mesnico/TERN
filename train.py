import pickle
import os
import time
import shutil
import yaml
import numpy as np

import torch
import pytorch_warmup as warmup

import data
from utils import get_model
from evaluation import t2i, AverageMeter, LogCollector, encode_data
from evaluate_utils.dcg import DCG

import logging
from torch.utils.tensorboard import SummaryWriter

import argparse


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', default='/w/31/faghri/vsepp_data/',
    #                     help='path to datasets')
    # parser.add_argument('--data_name', default='precomp',
    #                     help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    # parser.add_argument('--crop_size', default=224, type=int,
    #                     help='Size of an image crop as the CNN input.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads model, optimizer, scheduler')
    parser.add_argument('--load-model', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads only the model')
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--reinitialize-scheduler', action='store_true', help='Reinitialize scheduler. To use with --resume')
    parser.add_argument('--config', type=str, help="Which configuration to use. See into 'config' folder")

    opt = parser.parse_args()
    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger = SummaryWriter(log_dir=opt.logger_name, comment='')

    # Load data loaders
    train_loader, val_loader = data.get_loaders(
        config, opt.workers)

    # Construct the model
    model = get_model(config)
    if torch.cuda.is_available() and not (opt.resume or opt.load_model):
        model.cuda()

    # divide tern parameters from the bert ones, in order to have different learning rates during fine-tuning
    params, secondary_lr_multip = model.get_parameters()
    # validity check
    all_params = params[0] + params[1]
    if len(all_params) != len(list(model.parameters())):
        raise ValueError('Not all parameters are being returned! Correct get_parameters() method')

    if secondary_lr_multip > 0:
        optimizer = torch.optim.Adam([{'params': params[0]},
                                      {'params': params[1], 'lr': config['training']['lr']*secondary_lr_multip}],
                                     lr=config['training']['lr'])
    else:
        optimizer = torch.optim.Adam(params[0], lr=config['training']['lr'])

    # LR scheduler
    scheduler_name = config['training']['scheduler']
    if scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['training']['step-size'], gamma=config['training']['gamma'])
    elif scheduler_name is None:
        scheduler = None
    else:
        raise ValueError('{} scheduler is not available'.format(scheduler_name))

    # Warmup scheduler
    warmup_scheduler_name = config['training']['warmup'] if not opt.resume else None
    if warmup_scheduler_name == 'linear':
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=config['training']['warmup-period'])
    elif warmup_scheduler_name is None:
        warmup_scheduler = None
    else:
        raise ValueError('{} warmup scheduler is not available'.format(warmup_scheduler_name))

    # optionally resume from a checkpoint
    if opt.resume or opt.load_model:
        filename = opt.resume if opt.resume else opt.load_model
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            if torch.cuda.is_available():
                model.cuda()
            if opt.resume:
                start_epoch = checkpoint['epoch']
                best_rsum = checkpoint['best_rsum']
                optimizer.load_state_dict(checkpoint['optimizer'])
                if checkpoint['scheduler'] is not None and not opt.reinitialize_scheduler:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                # Eiters is used to show logs as the continuation of another
                # training
                model.Eiters = checkpoint['Eiters']
                print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                      .format(opt.resume, start_epoch, best_rsum))
            else:
                print("=> loaded only model from checkpoint '{}'"
                      .format(opt.load_model))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if torch.cuda.is_available():
        model.cuda()
    model.train()

    # load the ndcg scorer
    ndcg_val_scorer = DCG(config, len(val_loader.dataset), 'val', rank=25, relevance_methods=['rougeL', 'spice'])

    validate(val_loader, model, tb_logger, measure=config['training']['measure'], log_step=opt.log_step,
             ndcg_scorer=ndcg_val_scorer)

    # Train the Model
    best_rsum = 0
    best_ndcg = 0
    for epoch in range(opt.num_epochs):
        # train for one epoch
        train(opt, train_loader, model, optimizer, epoch, tb_logger, val_loader,
              measure=config['training']['measure'], grad_clip=config['training']['grad-clip'],
              scheduler=scheduler, warmup_scheduler=warmup_scheduler, ndcg_val_scorer=ndcg_val_scorer)

        # evaluate on validation set
        rsum, ndcg = validate(val_loader, model, tb_logger, measure=config['training']['measure'], log_step=opt.log_step,
                        ndcg_scorer=ndcg_val_scorer)

        # remember best R@ sum and save checkpoint
        is_best_rsum = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)

        is_best_ndcg = ndcg > best_ndcg
        best_ndcg = max(ndcg, best_ndcg)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'best_rsum': best_rsum,
            'best_ndcg': best_ndcg,
            'opt': opt,
            'config': config,
            'Eiters': model.Eiters,
        }, is_best_rsum, is_best_ndcg, prefix=opt.logger_name + '/')


def train(opt, train_loader, model, optimizer, epoch, tb_logger, val_loader, measure='cosine', grad_clip=-1, scheduler=None, warmup_scheduler=None, ndcg_val_scorer=None):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        model.train()
        if scheduler is not None:
            scheduler.step(epoch)

        if warmup_scheduler is not None:
            warmup_scheduler.dampen()

        optimizer.zero_grad()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        loss_dict = model(*train_data)
        loss = sum(loss for loss in loss_dict.values())

        # compute gradient and do SGD step
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.add_scalar('epoch', epoch, model.Eiters)
        tb_logger.add_scalar('step', i, model.Eiters)
        tb_logger.add_scalar('batch_time', batch_time.val, model.Eiters)
        tb_logger.add_scalar('data_time', data_time.val, model.Eiters)
        tb_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(val_loader, model, tb_logger, measure=measure, log_step=opt.log_step, ndcg_scorer=ndcg_val_scorer)


def validate(val_loader, model, tb_logger, measure='cosine', log_step=10, ndcg_scorer=None):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(
        model, val_loader, log_step, logging.info)

    # image retrieval
    (r1i, r5i, r10i, medri, meanr, mean_rougel_ndcg_i, mean_spice_ndcg_i) = t2i(
        img_embs, cap_embs, ndcg_scorer=ndcg_scorer, measure=measure)

    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, ndcg_rouge=%.4f ndcg_spice=%.4f" %
                 (r1i, r5i, r10i, medri, meanr, mean_rougel_ndcg_i, mean_spice_ndcg_i))
    # sum of recalls to be used for early stopping
    currscore = r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.add_scalar('r1i', r1i, model.Eiters)
    tb_logger.add_scalar('r5i', r5i, model.Eiters)
    tb_logger.add_scalar('r10i', r10i, model.Eiters)
    tb_logger.add_scalars('mean_ndcg_i', {'rougeL': mean_rougel_ndcg_i, 'spice': mean_spice_ndcg_i}, model.Eiters)
    tb_logger.add_scalar('medri', medri, model.Eiters)
    tb_logger.add_scalar('meanr', meanr, model.Eiters)
    tb_logger.add_scalar('rsum', currscore, model.Eiters)

    return currscore, mean_spice_ndcg_i


def save_checkpoint(state, is_best_rsum, is_best_ndcg, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best_rsum:
        shutil.copyfile(prefix + filename, prefix + 'model_best_rsum.pth.tar')
    if is_best_ndcg:
        shutil.copyfile(prefix + filename, prefix + 'model_best_ndcg.pth.tar')


if __name__ == '__main__':
    main()
