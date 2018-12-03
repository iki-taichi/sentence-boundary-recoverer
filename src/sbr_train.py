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
import json

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
        self.dataset_source = [
                'dataset/livedoor_corpus.json',
            ]
        # はじめにtrain, validに分ける直前だけで使う
        self.dataset_shuffle_random = np.random.RandomState(8310174)
        
        self.train_gpuid = 0
        self.train_save_each = 10
        self.train_max_sentence_len      = 1000
        self.train_minibacth_size        = 64
        self.train_minibacth_size_tuning = 256
        
        self.train_max_epoch   = 100
        self.train_schedule = {
                 0: {'phase': 'boundary'},
            }
        self.alpha=0.001 # Adam

        self.train_unk_rate  = 0.30
        
        self.gradient_clipping_threshold = 1.0
        
        self.read_ahead_len = 5
        self.max_sample_sentence_count = 1
        self.boundary_removing_rate = 0.95
        
        self.preload_model_path  = 'model_train/boundary_last.model'
        
    def make_model_name(self, namespace):
        import sys, re
        bn = os.path.basename(sys.argv[0])
        m = re.search('^' + re.escape(namespace) + '_(.+)\.py$', bn)
        if m is None:
            msg_tmp = 'file name must start with "%s_" and end with ".py"'
            raise Exception(msg_tmp%(namespace))
        return 'model_' + m.group(1)


def read_dataset_source(paths):
    blocks = []
    for path in paths:
        _blocks = []
        with open(path, 'r') as f:
            _blocks = json.load(f)
        _blocks = [b.strip() for b in _blocks]
        _blocks = [normalize_text(b) for b in _blocks]
        blocks.extend(_blocks)
    return blocks


def confuse_matrix(args, model, _dataset):
    _dataset.reset_iteration(
            mb_size=args.train_minibacth_size_tuning,
            max_sentence_len=args.train_max_sentence_len,
            max_sample_sentence_count=args.max_sample_sentence_count,
            boundary_removing_rate=args.boundary_removing_rate,
            postprocess=lambda x: model.xp.asarray(x),
        )
    
    choice_count = len(BOUNDARIES)
    mat = np.zeros((choice_count, choice_count), dtype='int32')
    
    with chainer.using_config('enable_backprop', False), \
            chainer.using_config('train', False):
        
        for _, mb_xs, mb_ys, mb_xs_text in _dataset:
            mb_size = mb_ys.shape[0]
            
            ps, _ = model.tf_loss(mb_xs, mb_ys)
            
            correct = mb_ys.flatten()
            prediction = ps.data.argmax(axis=1)
            
            for i, j in zip(correct, prediction):
                mat[int(i), int(j)] += 1
    
    print('# 第一成分:正解クラス, 第二成分:予測クラス')
    print(mat)
    exit()


def train(args):
    if not os.path.exists(args.dir_model):
        os.mkdir(args.dir_model)
    
    if args.train_gpuid >= 0:
        chainer.config.use_cudnn='always'
        chainer.cuda.get_device(args.train_gpuid).use()
    
    #
    # DATASET
    #
    blocks = read_dataset_source(args.dataset_source)
    args.dataset_shuffle_random.shuffle(blocks)
    len_train = int(len(blocks)*0.9)
    
    datasets = {}
    datasets['boundary_train'] = BoundaryDataset(
            blocks[:len_train], 'dynamic', 
            BOUNDARIES
        )
    datasets['boundary_tuning'] = BoundaryDataset(
            blocks[len_train:], 'static', BOUNDARIES
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
                special_tokens=sp_char_list,
            )
        boundary_model.embed.store_metadata(
                os.path.join(args.dir_model, 'embed_meta.pklb')
            )
    else:
        metadata_path = os.path.join(
                os.path.dirname(args.preload_model_path), 'embed_meta.pklb'
            )
        boundary_model = SentenceBoundaryModel(
                read_ahead_len=-1,
                boundaries=BOUNDARIES,
                metadata_path=metadata_path,
            )
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
        opts[model] = chainer.optimizers.Adam(alpha=args.alpha)
        opts[model].setup(model)
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
    
    # debug
    # confuse_matrix(args, models[0], datasets['boundary_tuning'])
    
    #
    # TRAINING LOOP
    #
    print('start training')
    for ep_id in range(args.train_max_epoch):
        tstart = time.time()
        args.ep_id = ep_id
        if ep_id in args.train_schedule:
            args.current_phase = args.train_schedule[ep_id]['phase']
        
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
    
    #print('epoch_boundary.train')
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
    
    #print('epoch_boundary.tuning')    
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
            blocks, 
            dataset_type, 
            boundaries, 
            random_state=None
        ):
        self.keys = ['_mb_pointer', 'xs', 'ys', '_xs_text']
        self.blocks = blocks
        self.dataset_type = dataset_type
        self.boundaries = boundaries
        self.random = random_state or np.random
        self.data_cache = None
        
    def description(self):
        s = [
                'Dataset Description:',
                '    type=%s'%'Boundary',
                '    len_blocks=%d'%len(self.blocks),
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
        [x] self.max_sample_sentence_count個, ブロックをサンプリングする
        [x] 結合する
        [x] 文字にラベルを振る
        [x] 確率で境界文字を消す
        """
        
        F_SURFACE=0
        F_NEXT_BOUNDARY_ID=1
        F_FILTER=2
        
        data_cache = []
        self.random.shuffle(self.blocks)
        i = 0
        
        while i < len(self.blocks):
            c = self.random.randint(1, self.max_sample_sentence_count+1)
            text = ''.join(self.blocks[i:i+c])
            char_info = [[t, 0, True] for t in text]
            # [surface, boundary_id, through filter]
            for j in range(len(text)-1):
                next_char_info = char_info[j+1]
                if (next_char_info[F_SURFACE] in self.boundaries) and \
                        self.random.uniform() < self.boundary_removing_rate:
                    char_info[j][F_NEXT_BOUNDARY_ID] = \
                            self.boundaries.index(next_char_info[F_SURFACE])
                    next_char_info[F_FILTER] = False
            char_info = list(filter(lambda _: _[F_FILTER], char_info))
            char_info = char_info[:self.max_sentence_len]
            xs = self.ts2is_in(''.join([_[F_SURFACE] for _ in char_info]))
            ys = [_[F_NEXT_BOUNDARY_ID] for _ in char_info]
            
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

        #print_count = 0
        
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
            
            #print_count -= 1
            #if print_count <= 0:
            #    print_count = 100
            #    print(mb_pointer, total_mb, total_correct_ans/float(total_mb))
        
        loss_mean = loss_sum / loss_sum_count
        acc = total_correct_ans/float(total_mb)
        return loss_mean, acc
    

if __name__ == '__main__':
    args = ArgSpace()
    train(args)

