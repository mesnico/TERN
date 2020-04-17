from __future__ import print_function

import numpy

from data import get_test_loader
import time
import numpy as np
import torch
from collections import OrderedDict
from utils import get_model
from evaluate_utils.dcg import DCG
from models.loss import order_sim


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.add_scalar(prefix + k, v.val, global_step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    for i, (images, targets, img_lengths, cap_lengths, boxes, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        if type(targets) == tuple or type(targets) == list:
            captions, features, wembeddings = targets
            # captions = features  # Very weird, I know
            text = features
        else:
            text = targets
            captions = targets
            wembeddings = model.img_txt_enc.txt_enc.word_embeddings(captions.cuda() if torch.cuda.is_available() else captions)

        # compute the embeddings
        with torch.no_grad():
            img_emb, cap_emb, _, _ = model.forward_emb(images, text, img_lengths, cap_lengths, boxes)

            # initialize the numpy arrays given the size of the embeddings
            if img_embs is None:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

            # preserve the embeddings by copying from gpu and converting to numpy
            img_embs[ids, :] = img_emb.data.cpu().numpy().copy()
            cap_embs[ids, :] = cap_emb.data.cpu().numpy().copy()

            # measure accuracy and record loss
            model.forward_loss(img_emb, cap_emb)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions

    # p = np.random.permutation(len(data_loader.dataset) // 5) * 5
    # p = np.transpose(np.tile(p, (5, 1)))
    # p = p + np.array([0, 1, 2, 3, 4])
    # p = p.flatten()
    # img_embs = img_embs[p]
    # cap_embs = cap_embs[p]

    return img_embs, cap_embs


def evalrank(config, checkpoint, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options

    # construct model
    model = get_model(config)

    # load model state
    model.load_state_dict(checkpoint['model'], strict=False)

    print('Loading dataset')
    data_loader = get_test_loader(config, vocab=None, workers=4, split_name=split)

    # initialize ndcg scorer
    ndcg_val_scorer = DCG(config, len(data_loader.dataset), split, rank=25, relevance_methods=['rougeL', 'spice'])

    print('Computing results...')
    img_embs, cap_embs = encode_data(model, data_loader)

    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        ri, rti = t2i(img_embs, cap_embs, return_ranks=True, ndcg_scorer=ndcg_val_scorer)
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f, ndcg_rouge=%.4f, ndcg_spice=%.4f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000], cap_embs[i * 5000:(i + 1) * 5000],
                           return_ranks=True, ndcg_scorer=ndcg_val_scorer, fold_index=i)
            if i == 0:
                rti = rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, ndcg_rouge=%.4f, ndcg_spice=%.4f" % ri)
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ari: %.1f" % (rsum, ari))
            results += [list(ri) + [ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[8] * 6))

        print("Average t2i Recall: %.1f" % mean_metrics[7])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f ndcg_rouge=%.4f ndcg_spice=%.4f" %
              mean_metrics[:7])


def t2i(images, captions, npts=None, return_ranks=False, ndcg_scorer=None, fold_index=0, measure='dot'):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    top50 = numpy.zeros((5 * npts, 50))
    rougel_ndcgs = numpy.zeros(5 * npts)
    spice_ndcgs = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]
        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = numpy.dot(queries, ims.T)

        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][
                0]  # in che posizione e' l'immagine (index) che ha questa caption (5*index + i)
            top50[5 * index + i] = inds[i][0:50]
            # calculate ndcg
            if ndcg_scorer is not None:
                rougel_ndcgs[5 * index + i], spice_ndcgs[5 * index + i] = \
                    ndcg_scorer.compute_ndcg(npts, 5 * index + i, inds[i].astype(int),
                                             fold_index=fold_index, retrieval='image').values()

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    mean_rougel_ndcg = np.mean(rougel_ndcgs)
    mean_spice_ndcg = np.mean(spice_ndcgs)

    if return_ranks:
        return (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg), (ranks, top50)
    else:
        return (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg)

