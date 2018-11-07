# coding: utf-8

"""
sbr_train.py

Written_by: Taichi Iki (taici.iki@gmail.com)
Created_at: 2018-09-24
Comment:
日本語文の文境界復活モデル
    - 学習用スクリプト

学習用のため本スクリプトのコメントは省略気味
学習開始方法
    python src/sbr_train.py
ArgSpace内で設定を変える
"""


import os
import time
import re
import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
import chainer.variable as V
from chainer.optimizer import GradientClipping
from chainer.optimizer import WeightDecay
import cupy

from sbr import \
        SYMBOL_BOS, SYMBOL_EOS, SYMBOL_UNK, BOUNDARIES, \
        normalize_text, SentenceBoundaryModel


class ArgSpace(object):
    """
    学習実行時の設定をまとめるクラス
    """
    
    def __init__(self, dir_model=None):
        # model namespace
        self.namespace = 'sbr'
        
        self.dir_model = dir_model
        if self.dir_model is None:
            self.dir_model = self.make_model_name(self.namespace)
        
        # dataset
        # データセットは一行に1文のutf-8テキスト
        self.dataset_sentences = [
                'dataset/nuc_no_header.txt',
            ]
        
        self.train_gpuid = 1
        self.train_save_each = 5
        self.train_max_sentence_len      = 1000
        self.train_minibacth_size        = 16
        self.train_minibacth_size_tuning = 16
        
        self.train_max_epoch   = 30
        self.train_schedule = {
                 0: {'phase': 'boundary', 'lr': 1.0},
                 10: {'phase': 'boundary', 'lr': 0.5},
                 20: {'phase': 'boundary', 'lr': 0.25},
            }

        self.train_unk_rate  = 0.30
        
        self.weight_decay_rate = 1.0e-6
        self.gradient_clipping_threshold = 1.0
        
        self.read_ahead_len = 5
        self.max_sample_sentence_count = 5
        self.boundary_removing_rate = 0.80
        
        self.preload_model_path  = None
        
    def make_model_name(self, namespace):
        import sys, re
        bn = os.path.basename(sys.argv[0])
        m = re.search('^' + re.escape(namespace) + '_(.+)\.py$', bn)
        if m is None:
            msg_tmp = 'file name must start with "%s_" and end with ".py"'
            raise Exception(msg_tmp%(namespace))
        return 'model_' + m.group(1)


def read_sentences(paths):
    sentences = []
    for path in paths:
        lines = []
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        lines = [normalize_text(l) for l in lines]
        sentences.extend(lines)
    return lines


def train(args):
    if not os.path.exists(args.dir_model):
        os.mkdir(args.dir_model)
    
    if args.train_gpuid >= 0:
        chainer.config.use_cudnn='always'
        chainer.cuda.get_device(args.train_gpuid).use()
    
    #
    # DATASET
    #
    sentences = read_sentences(args.dataset_sentences)
    np.random.shuffle(sentences)
    len_train = int(len(sentences)*0.8)
    
    datasets = {}
    datasets['boundary_train'] = BoundaryDataset(
            sentences[:len_train], 'dynamic', 
            BOUNDARIES
        )
    datasets['boundary_tuning'] = BoundaryDataset(
            sentences[len_train:], 'static', BOUNDARIES
        )
    for ds in datasets.values():
        print(ds.description())
    
    #
    # MODELS
    #
    models = []
    
    sp_char_list = [SYMBOL_UNK, SYMBOL_BOS, SYMBOL_EOS]
    embed_path = 'data/sg_d128_w8_mc0_neg5_iter10_s0p001.pklb'
    
    # boundary model
    if args.preload_model_path is None:
        boundary_model = SentenceBoundaryModel(
                read_ahead_len=args.read_ahead_len,
                boundaries=BOUNDARIES,
                embed_path=embed_path,
                special_tokens=sp_char_list
            )
        boundary_model.embed.store_metadata(
                os.path.join(args.dir_model, 'embed_meta.pklb')
            )
    else:
        metadata_path = os.path.join(
                os.path.dirname(args.preload_model_path), 'embed_meta.pklb'
            )
        boundary_model = SentenceBoundaryModel(metadata_path=metadata_path)
        chainer.serializers.load_npz(args.preload_model_path, boundary_model)
    models.append(boundary_model)
    
    if args.train_gpuid >= 0:
        for model in models:
            model.to_gpu(args.train_gpuid)
    
    def save_models(save_dir, postfix):
        for model in models:
            save_path = os.path.join(save_dir, model.elem_name + postfix)
            model.to_cpu()
            chainer.serializers.save_npz(save_path, model)
            if args.train_gpuid >= 0:
                model.to_gpu(args.train_gpuid)
    
    #
    # OPIMIZERS
    #
    opts = {}
    for model in models:
        opts[model] = chainer.optimizers.SGD(lr=1.0)
        opts[model].setup(model)
        opts[model].add_hook(
                WeightDecay(rate=args.weight_decay_rate)
            )
        opts[model].add_hook(
                GradientClipping(threshold=args.gradient_clipping_threshold)
            )
    
    #
    # DICTIONARY REGISTRATION
    #
    print('token dictionary size:', len(boundary_model.token2id))

    datasets['boundary_train'].register_token_mapping(
            lambda xs: boundary_model.ts2is_with_unk(xs, np.random, args.train_unk_rate)
        )
    datasets['boundary_tuning'].register_token_mapping(boundary_model.ts2is)
    
    #
    # TRAINING LOOP
    #
    print('start training')
    for ep_id in range(args.train_max_epoch):
        tstart = time.time()
        args.ep_id = ep_id
        if ep_id in args.train_schedule:
            args.current_lr = args.train_schedule[ep_id]['lr']
            args.current_phase = args.train_schedule[ep_id]['phase']
            for opt in opts.values():
                opt.lr = args.current_lr
        
        current_log = ['ep=%d phase=%s'%(args.ep_id, args.current_phase)]
        
        # rooting training types
        if args.current_phase == 'boundary':
            epoch_boundary(args, models, opts, datasets, current_log)
        
        # Writing log lines
        current_log.append('elapsed=%d'%(time.time() - tstart))
        log_string = ' '.join(current_log)
        print(log_string)
        with open(os.path.join(args.dir_model, 'log_summary.txt'), 'a') as f:
            f.write(log_string+'\n')
        
        # Storing Trained Model
        if ep_id % args.train_save_each == 0:
            save_models(args.dir_model, '_ep_%d.model'%(ep_id))
            print('model saved.')
        
    print('training done.')
    
    # Storing the Last Model
    save_models(args.dir_model, '_last.model')
    print('model saved.')


def epoch_boundary(args, models, opts, datasets, current_log):
    """
    boundaryフェーズの学習
    """

    model = models[0]
    opt = opts[model]
    ds_train = datasets['boundary_train']
    ds_tuning = datasets['boundary_tuning']

    # Train
    ds_train.reset_iteration(
            mb_size=args.train_minibacth_size,
            max_sentence_len=args.train_max_sentence_len,
            max_sample_sentence_count=args.max_sample_sentence_count,
            boundary_removing_rate=args.boundary_removing_rate,
            postprocess=lambda x: model.xp.asarray(x),
        )
    
    print('epoch_boundary.train')
    with chainer.using_config('enable_backprop', True), \
            chainer.using_config('train', True):
        train_loss_mean, train_acc = ds_train.pour(args, model, opt)
        current_log.append('train_loss=%.3f'%train_loss_mean)
        current_log.append('train_acc=%.3f'%train_acc)
            
    # Tuning(evaluating)
    ds_tuning.reset_iteration(
            mb_size=args.train_minibacth_size_tuning,
            max_sentence_len=args.train_max_sentence_len,
            max_sample_sentence_count=args.max_sample_sentence_count,
            boundary_removing_rate=args.boundary_removing_rate,
            postprocess=lambda x: model.xp.asarray(x),
        )
    
    print('epoch_boundary.tuning')    
    with chainer.using_config('enable_backprop', False), \
            chainer.using_config('train', False):
        eval_loss_mean, eval_acc = ds_tuning.pour(args, model, None)
        current_log.append('eval_loss=%.3f'%eval_loss_mean)
        current_log.append('eval_acc=%.3f'%eval_acc)


class BoundaryDataset(object):
    """
    文境界復元用のデータセットと学習の定義
    """
    
    def __init__(
            self, 
            sentences, 
            dataset_type, 
            boundaries, 
            random_state=None
        ):
        self.keys = ['_mb_pointer', 'xs', 'ys', '_xs_text']
        self.sentences = sentences
        self.dataset_type = dataset_type
        self.boundaries = boundaries
        self.random = random_state or np.random
        self.data_cache = None
        
    def description(self):
        s = [
                'Dataset Description:',
                '    type=%s'%'Boundary',
                '    len_sentences=%d'%len(self.sentences),
            ]
        return '\n'.join(s)

    def register_token_mapping(self, ts2is_in):
        self.ts2is_in  = ts2is_in

    def reset_iteration(self, **kwargs):
        """
        keyword arguments
            mb_size, 
            max_sentence_len, 
            max_sample_sentence_count, 
            boundary_removing_rate,
            postprocess
        """
        
        for k,v in kwargs.items():
            setattr(self, k, v)

        if self.data_cache is None:
            self.data_cache = self._make_data_cache()
        
        self.data_mapping = list(range(0, len(self.data_cache)))
        self.random.shuffle(self.data_mapping)
        self.iter_range = iter(range(0, len(self.data_cache), self.mb_size))
    
    def _make_data_cache(self):
        """
        [x] self.max_sample_sentence_count個, 文をサンプリングする
        [x] 結合する
        [x] 文字にラベルを振る
        [x] 確率で境界文字を消す
        """
        
        data_cache = []
        self.random.shuffle(self.sentences)
        i = 0
        while i < len(self.sentences):
            c = self.random.randint(1, self.max_sample_sentence_count+1)
            text = ''.join(self.sentences[i:i+c])
            buf = [[t, 0, True] for t in text]
            # [surface, boundary_id, through filter]
            for j in range(len(text)-1):
                next_char = buf[j+1][0]
                if (next_char in self.boundaries) and \
                        self.random.uniform() < self.boundary_removing_rate:
                    buf[j][1] = self.boundaries.index(next_char)
                    buf[j+1][2] = False
            buf = list(filter(lambda t: t[2], buf))
            buf = buf[:self.max_sentence_len]
            xs = self.ts2is_in(''.join([_[0] for _ in buf]))
            ys = [_[1] for _ in buf]
            
            data_cache.append({
                    'xs': xs,
                    'ys': ys,
                    '_xs_text': text,
                })
             
            i+=c
    
        return data_cache

    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            i = next(self.iter_range)
            self.current_index = i
        except StopIteration as e:
            if self.dataset_type != 'static':
                self.data_cache = None
            raise e

        return self.make_minibatch(self.current_index)
    
    def make_minibatch(self, mb_pointer):
        ids = self.data_mapping[mb_pointer:mb_pointer+self.mb_size]
        mb = [self.data_cache[_] for _ in ids]
        
        ret = [mb_pointer]
        
        for key in self.keys[1:]:
            seqs = [_[key] for _ in mb]
            
            if key in ['xs', 'ys']:
                length = [len(seq) for seq in seqs]
                m = max(length)
                x = -1 * np.ones((len(seqs), m), dtype='int32')
                for j in range(len(seqs)):
                    mm = min(length[j], m)
                    x[j][:mm] = np.asarray(seqs[j], dtype='int32')[:mm]
            
            elif key.startswith('_'):
                x = seqs
            
            ret.append(x)
            
        if not(self.postprocess is None):
            for j in range(len(self.keys)):
                if not self.keys[j].startswith('_'):
                    ret[j] = self.postprocess(ret[j])
        
        return ret

    def pour(self, args, model, opt):
        """
        modelに関するデータセット一周分の学習
        """

        total_mb          = 0
        total_correct_ans = 0
        loss_sum       = 0
        loss_sum_count = 0

        print_count = 0
        
        for mb_pointer, mb_xs, mb_ys, mb_xs_text in self:
            mb_size = mb_ys.shape[0]
        
            ps, loss = model.tf_loss(mb_xs, mb_ys)
        
            if chainer.config.enable_backprop:
                model.zerograds()
                loss.backward()
                opt.update()
                loss.unchain_backward()
        
            ps = F.sigmoid(ps.data).data
            predicted_labels = ps.argmax(axis=1)
        
            total_mb += (mb_ys != -1).sum()
            total_correct_ans += int(
                    (predicted_labels == mb_ys.reshape((-1,))).sum()
                )
            loss_sum       += float(loss.data)*mb_size
            loss_sum_count += mb_size
            
            print_count -= 1
            if print_count <= 0:
                print_count = 100
                print(mb_pointer, total_mb, total_correct_ans/float(total_mb))
        
        loss_mean = loss_sum / loss_sum_count
        acc = total_correct_ans/float(total_mb)
        return loss_mean, acc
    

if __name__ == '__main__':
    args = ArgSpace()
    train(args)

