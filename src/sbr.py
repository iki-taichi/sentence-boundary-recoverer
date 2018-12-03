# coding: utf-8

"""
sbr.py

Written_by: Taichi Iki (taici.iki@gmail.com)
Created_at: 2018-09-24
Comment:
日本語文の文境界復活モデル
    - モデル定義
    - モデルラッパー
"""


import pickle
import os

import chainer.cuda as cuda
import chainer.functions as F
import chainer.links as L
import chainer

import mojimoji

try:
    from src.jaembed import JaEmbedID
except:
    from jaembed import JaEmbedID


SYMBOL_BOS = '<BOS>'
SYMBOL_EOS = '<EOS>'
SYMBOL_UNK = '<UNK>'

BOUNDARIES = [
        None,
        '、',
        '。',
        '!',
        '?',
        '\n',
    ]


def normalize_text(text):
    """
    入力テキストの正規化
    """

    text = mojimoji.zen_to_han(text, kana=False, digit=True, ascii=True)
    text = mojimoji.han_to_zen(text, kana=True, digit=False, ascii=False)
    # 暫定処理
    text.replace('.', '。')
    text.replace(',', '、')
    return text


class BoundaryRecoverer(object):
    """
    文境界復元モデルのラッパー
    使用法:
        br = BoundaryRecoverer()
        br('境界を復元したい文字列を入れる')
    """
    
    def __init__(
            self, 
            model_path=None,
            max_sentence_len=10000,
            gpu_id=-1
        ):
        """
        model_path: None or 学習済みモデルのパス. Noneの時はデフォルトを探す．
        max_sentence_len: 最大で受け付ける文字数
        gpu_id: GPUでモデルを走らせる場合のGPUのID(CPUの場合は-1)
        """

        self.model_path = model_path or self.get_model_path()
        self.max_sentence_len = max_sentence_len
        self.gpu_id = gpu_id

        if self.gpu_id >= 0:
            chainer.config.use_cudnn='always'
            cuda.get_device(self.gpu_id).use()
        
        metadata_path = os.path.join(
                os.path.dirname(self.model_path), 
                'embed_meta.pklb'
            )
        self.model = SentenceBoundaryModel(
                read_ahead_len=-1,
                boundaries=BOUNDARIES,
                metadata_path=metadata_path,
            )
        chainer.serializers.load_npz(self.model_path, self.model)
        self.read_ahead_len = self.model.read_ahead_len
        
        if self.gpu_id >= 0:
            model.to_gpu(self.gpu_id)
    
    def get_model_path(self):
        """
        __file__からモデルパスを推測
        """

        d = os.path.dirname(__file__)
        for t in range(2):
            p = os.path.join(d, 'model_train/boundary_last.model')
            if os.path.exists(p):
                return p
            d = os.path.dirname(d)
        raise Exception('No trained model found.')
    
    def __call__(self, text):
        """
        textに対して境界推定を実施
        """

        xp = self.model.xp
        
        text = text[:self.max_sentence_len]
        xs = xp.asarray(
                self.model.ts2is(normalize_text(text)), 
                dtype='int32'
            )[None,:]
        
        buf = []
        with chainer.using_config('enable_backprop', False), \
                chainer.using_config('train', False):
            state = self.model.get_init_state(xs[:,:self.read_ahead_len-1])
            for i in range(0, len(text)):
                p, state = self.model.prob(
                        state, 
                        xs[:,self.read_ahead_len+i-1:self.read_ahead_len+i]
                    )
                buf.append((text[i], int(p.data[0].argmax())))

        ret = ''
        for t in buf:
            ret += t[0]
            if t[1] != 0:
                ret += BOUNDARIES[t[1]]

        return ret


class SentenceBoundaryModel(chainer.Chain):
    def __init__(
            self, 
            read_ahead_len, 
            boundaries, 
            metadata_path=None, 
            embed_path=None, 
            special_tokens=None
        ):
        """
        read_ahead_len: 先読み数(モデルを読み込む時はダミーで良い)
        boundaries: 推測する文境界の文字種のリスト
        metadata_path: None or jaembedのメタファイルのパス
        embed_path: None or 事前学習した埋め込みのパス
        special_tokens: 事前学習に含まれないトークン
        """
        super(SentenceBoundaryModel, self).__init__()
        self.elem_name = 'boundary'
        
        self.add_persistent('read_ahead_len', read_ahead_len)
        self.boundaries = boundaries
        
        self.ignore_label = -1
        self.dropout_ratio=0.5
        
        self.charvec_dim = 128
        self.encode_layer_num = 1
        self.encode_dim = 512
        
        HeN = chainer.initializers.HeNormal
        with self.init_scope():
            self.embed = JaEmbedID(
                    embed_path, special_tokens, metadata_path, 
                    initialW=HeN(fan_option='fan_out')
                )
            
            self.encode_lstm = L.NStepLSTM(
                    self.encode_layer_num, 
                    self.charvec_dim, 
                    self.encode_dim, 
                    dropout=self.dropout_ratio
                )
            
            self.output_fc = L.Linear(
                    self.encode_dim,
                    len(self.boundaries),
                    initialW=HeN(),
                    nobias=True
                )
        
        self.token2id = self.embed.token2id
        self.char_eos_id = self.token2id[SYMBOL_EOS]
        self.char_unk_id = self.token2id[SYMBOL_UNK]
        

    def tf_loss(self, xs, ys):
        """
        teacher forcingでロスを計算する(学習時)
        arguments:
            xs: 文字列(各系列の終わりはread_ahed_len-1回の<EOS>であること)
            ys: 正解ラベル(系列のi番目はi文字の次の境界を予測)
        """
        
        mb_size, max_step = xs.shape
        masks = (xs != self.ignore_label)
        steps = masks.sum(axis=1)
        assert (steps != 0).all(), '0 step sentence was given.'
        
        emb_xs = self.embed(xs)
        emb_xs = F.dropout(emb_xs, self.dropout_ratio)
        
        _, _, hs = self.encode_lstm(
                hx=None, 
                cx=None, 
                xs=[v[:l,:] for l, v in zip(steps, emb_xs)]
            )
        hs = F.pad_sequence(hs, length=max_step, padding=-10)
        hs = F.reshape(hs[:, self.read_ahead_len-1:, :], (-1, hs.shape[-1]))
        # hs: list(steps, dim) -> (mb_size*(step_max-read_ahead_len+1), dim)
        
        ps = self.output_fc(hs)
        loss = F.softmax_cross_entropy(ps, self.xp.reshape(ys, (-1,)))
        
        return ps, loss
    
    def get_init_state(self, xs):
        """
        初めのself.read_ahead_len-1文字からstateを作る
        arguments:
            xs: self.read_ahead_len-1文字
        """

        assert xs.shape[1] == self.read_ahead_len-1, \
            'initial input len should be self.read_ahead_len-1.'
            
        emb_xs = self.embed(xs[:,:-1])
        h, c, hs = self.encode_lstm(hx=None, cx=None, xs=list(emb_xs))
        return (h, c)
    
    def prob(self, state, xs):
        """
        入力された文字に続く文境界を予測する
        arguments:
            state: (h, c)
            xs: 1文字
        """
        
        h, c = state
        emb_xs = self.embed(xs)
        
        # ステップの次元がなければ追加
        if len(emb_xs.shape) == 2:
            emb_xs = emb_xs[:,None,:]
        
        h, c, hs = self.encode_lstm(hx=h, cx=c, xs=list(emb_xs))
        hs = F.pad_sequence(hs, length=1, padding=-10)
        hs = F.reshape(hs, (-1, hs.shape[-1]))
        
        ps = self.output_fc(hs)
        
        return ps, (h, c)
    
    def ts2is(self, ts):
        """
        文字列をidリストに変換する
        """

        return \
            [self.token2id.get(t, self.char_unk_id) for t in ts] + \
            [self.char_eos_id]*(self.read_ahead_len-1)
    
    def ts2is_with_unk(self, ts, random, rate):
        """
        文字列をidリストに変換する(ドロップノイズ有り)
        """

        force_unk = random.uniform(size=len(ts)) < rate
        return \
            [self.token2id.get(None if fu else t, self.char_unk_id) \
            for fu, t in zip(force_unk, ts)] + \
            [self.char_eos_id]*(self.read_ahead_len-1)

