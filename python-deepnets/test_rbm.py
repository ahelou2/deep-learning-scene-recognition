import sys
import time
import numpy
import glumpy
import pickle
import logging
import datetime
import optparse
import collections
import numpy.random as rng

from PIL import Image
from OpenGL import GL as gl

import rbm
import idx_reader

FLAGS = optparse.OptionParser()
FLAGS.add_option('', '--model')
FLAGS.add_option('-i', '--images')
FLAGS.add_option('-l', '--labels')
FLAGS.add_option('-f', '--frames', action='store_true')
FLAGS.add_option('-b', '--binary', action='store_true')
FLAGS.add_option('-a', '--alpha', type=float, default=0.5)
FLAGS.add_option('-t', '--tau', type=float, default=500.)
FLAGS.add_option('-m', '--momentum', type=float, default=0.2)
FLAGS.add_option('', '--l2', type=float, default=0.001)
FLAGS.add_option('-p', '--sparsity', type=float, default=0.)
FLAGS.add_option('-r', '--rows', type=int, default=10)
FLAGS.add_option('-c', '--cols', type=int, default=10)
FLAGS.add_option('-s', '--batch-size', type=int, default=20)


def now():
    return datetime.datetime.now()


def save_frame(width, height):
    pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    Image.fromstring(mode="RGB", size=(width, height), data=pixels
                     ).transpose(Image.FLIP_TOP_BOTTOM
                                 ).save(now().strftime('frame-%Y%m%d-%H%M%S.%f.png'))


def mean(x):
    return sum(x) / max(1, len(x))


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='%(levelname).1s %(asctime)s [%(module)s:%(lineno)d] %(message)s')

    opts, args = FLAGS.parse_args()

    visibles = numpy.zeros((opts.rows // 2, 28, 28), 'f')
    hiddens = numpy.zeros((opts.rows // 2, opts.rows, opts.cols), 'f')
    weights = numpy.zeros((opts.rows, opts.cols, 28, 28), 'f')

    m = opts.binary and 1.5 or 0.5
    kwargs = dict(cmap=glumpy.colormap.Grey, vmin=0, vmax=255)
    _visibles = [glumpy.Image(v, **kwargs) for v in visibles]
    _hiddens = [glumpy.Image(h) for h in hiddens]
    _weights = [[glumpy.Image(w, vmin=-m, vmax=m) for w in ws] for ws in weights]

    W = 100 * (opts.cols + 1) + 4
    H = 100 * opts.rows + 4

    win = glumpy.Window(W, H)

    loaded = False
    updates = -1
    batches = 0.
    recent = collections.deque(maxlen=20)
    errors = [collections.deque(maxlen=20) for _ in range(10)]
    testset = [None] * 10
    trainset = dict((i, []) for i in range(10))
    loader = idx_reader.iterimages(opts.labels, opts.images, False)

    rbm = opts.model and pickle.load(open(opts.model, 'rb')) or rbm.RBM(
        28 * 28, opts.rows * opts.cols, opts.binary)
 
    trainer = rbm.Trainer(rbm,
                          momentum=opts.momentum,
                          target_sparsity=opts.sparsity,
                          )

    def get_pixels():
        global loaded
        if not loaded and numpy.all([len(trainset[t]) > 10 for t in range(10)]):
            loaded = True

        if loaded and rng.random() < 0.99:
            t = rng.randint(10)
            pixels = trainset[t][rng.randint(len(trainset[t]))]
        else:
            t, pixels = loader.next()
            if testset[t] is None and rng.random() < 0.3:
                testset[t] = pixels
                raise RuntimeError
            else:
                trainset[t].append(pixels)

        recent.append(pixels)
        if len(recent) < 20:
            raise RuntimeError

        return pixels

    def flatten(pixels):
        if opts.binary:
            return pixels.reshape((1, 28 * 28)) > 30.
        r = numpy.array(recent)
        mu = r.mean(axis=0)
        sigma = numpy.clip(r.std(axis=0), 0.1, numpy.inf)
        return ((pixels - mu) / sigma).reshape((1, 28 * 28))

    def unflatten(flat):
        if opts.binary:
            return 256. * flat.reshape((28, 28))
        r = numpy.array(recent)
        mu = r.mean(axis=0)
        sigma = r.std(axis=0)
        return sigma * flat.reshape((28, 28)) + mu

    def learn():
        global batches

        batch = numpy.zeros((opts.batch_size, 28 * 28), 'd')
        for i in range(opts.batch_size):
            while True:
                try:
                    pixels = get_pixels()
                    break
                except RuntimeError:
                    pass
            flat = flatten(pixels)
            batch[i:i+1] = flat

        batches += 1
        trainer.learn(batch, alpha=opts.alpha * numpy.exp(-batches / opts.tau), l2_reg=opts.l2)

        logging.info('mean weight: %.3g, vis bias: %.3g, hid bias: %.3g',
                     rbm.weights.mean(), rbm.vis_bias.mean(), rbm.hid_bias.mean())

        return pixels, flat

    def update(pixels, flat):
        for i, (v, h) in enumerate(rbm.iter_passes(flat)):
            if i == len(visibles):
                break
            visibles[i] = unflatten(v)
            hiddens[i] = h.reshape((opts.rows, opts.cols))
        [v.update() for v in _visibles]
        [h.update() for h in _hiddens]

        for r in range(opts.rows):
            for c in range(opts.cols):
                weights[r, c] = rbm.weights[r * opts.cols + c].reshape((28, 28))
        [[w.update() for w in ws] for ws in _weights]

    def evaluate():
        for t, pixels in enumerate(testset):
            if pixels is None:
                continue
            estimate = unflatten(rbm.reconstruct(flatten(pixels)))
            errors[t].append(((pixels - estimate) ** 2).mean())
        report = ' : '.join('%d' % mean(errors[t]) for t in range(10))
        r = numpy.array(recent)
        logging.error('%d<%.3g>: %.3g+%.3g: %s', batches,
                      opts.alpha * numpy.exp(-batches / opts.tau),
                      r.mean(axis=0).mean(), r.std(axis=0).mean(), report)

    @win.event
    def on_draw():
        p = 4
        W, H = win.get_size()
        w = int(float(W - p) / (opts.cols + 1))
        h = int(float(H - p) / opts.rows)
        i = 0
        for vis, hid in zip(_visibles, _hiddens):
            vis.blit(p, h * i + p, w - p, h - p)
            i += 1
            hid.blit(p, h * i + p, w - p, h - p)
            i += 1
        for r in range(opts.rows):
            ws = _weights[r]
            for c in range(opts.cols):
                ws[c].blit(w * (c + 1) + p, h * r + p, w - p, h - p)

    @win.event
    def on_idle(dt):
        global updates
        if updates:
            updates -= 1
            update(*learn())
            win.draw()
            if opts.frames:
                save_frame(W, H)
            evaluate()

    @win.event
    def on_key_press(key, modifiers):
        global updates
        if key == glumpy.key.ESCAPE:
            sys.exit()
        if key == glumpy.key.S:
            pickle.dump(rbm, open(now().strftime('rbm-%Y%m%d-%H%M%S.p'), 'wb'))
        if key == glumpy.key.SPACE:
            updates = updates == 0 and -1 or 0
        if key == glumpy.key.ENTER:
            if updates >= 0:
                updates = 1

    win.mainloop()
