# coding: utf-8


"""
jaembed.py

Written_by: Taichi Iki (taici.iki@gmail.com)
Created_at: 2018-11-07
Comments:
JaEmbded - a Japanese Character Embedding for Chainer
Requires chainer library
"""


import chainer
import chainer.cuda as cuda
import chainer.functions as F
import chainer.links as L

import pickle


class JaEmbedID(chainer.Chain):
    """
    事前学習した文字埋め込みと特殊文字の文字埋め込みをハイブリッドで使う
    """
    
    def to_gpu(self, device=None):
        self.W = cuda.to_gpu(self.W, device)
        super(JaEmbedID, self).to_gpu(device)
    
    def to_cpu(self):
        self.W = cuda.to_cpu(self.W)
        super(JaEmbedID, self).to_cpu()
    
    def __init__(self, 
            base_data_path, 
            special_tokens, 
            metadata_path=None, 
            initialW=None
        ):
        """
        Arguments
            base_data_path: None or 学習済みベクトルへのパス
            special_tokens: [] or 学習可能トークン
            metadata_path: None or 学習結果のメタデータパス
            initialW: 初期化の方法
        """

        super(JaEmbedID, self).__init__()
        
        self.ignore_label = -1
        self.base_data_path = base_data_path
        
        if not (self.base_data_path is None):
            if self.base_data_path.endswith('.pklb'):
                tl, tv = self.load_vectors_from_pklb(self.base_data_path)
            elif self.base_data_path.endswith('.bin'):
                tl, tv = self.load_vectors_from_bin(self.base_data_path)
            else:
                raise Exception('unknown format for embedding base')
        
            self.token_list = tl
            self.add_persistent('W', self.xp.asarray(tv, dtype='float32'))
            self.vector_dim = tv.shape[1]
            
            self.normal_token_len = len(self.token_list)
            self.token_list.extend(special_tokens)
            self.total_token_len = len(self.token_list)
            self.special_token_len = \
                    self.total_token_len - self.normal_token_len
        
        if not (metadata_path is None):
            self.restore_metadata(metadata_path)
            # An empty matrix allocated for model-loading after initializing.
            W_shape = (self.normal_token_len, self.vector_dim)
            self.add_persistent('W', self.xp.zeros(W_shape, dtype='float32'))
        
        with self.init_scope():
            self.embed_special = L.EmbedID(
                    self.special_token_len, 
                    self.vector_dim, 
                    ignore_label=self.ignore_label, 
                    initialW=initialW
                )
        
        self.token2id = self.make_dict()
    
    def load_vectors_from_pklb(self, path):
        """
        バイナリーのpickleから元となる学習済みトークンベクトルを読み込む
        Arguments:
            path: バイナリーのpickleへのパス
        """
        d = None
        with open(path, 'rb') as f:
            d = pickle.Unpickler(f).load()
        return d['charlist'], d['charvecs']

    def load_vectors_from_bin(self, path):
        """
        gensimのword2vecのbinから元となる学習済みトークンベクトルを読み込む
        Arguments:
            path: gensimのword2vecのbinへのパス
        """
        from gensim.models import KeyedVectors
        wv = KeyedVectors.load_word2vec_format(path, binary=True)
        tl = list(sorted(wv.vocab.keys()))
        tv = np.asarray([wv[token] for token in tl])
        return tl, tv
    
    def store_metadata(self, path):
        """
        学習後にモデルを再生できるよう辞書等をメタファイルとして保存する
        Arguments:
            path: 保存先のパス
        """
        metadata={
                'token_list': self.token_list,
                'normal_token_len': self.normal_token_len,
                'vector_dim': self.vector_dim,
            }
        with open(path, 'wb') as f:
            pickle.Pickler(f).dump(metadata)

    def restore_metadata(self, path):
        """
        辞書等をメタファイルを使ってモデル情報を復元する
        Arguments:
            path: 保存先のパス
        """
        metadata = {}
        with open(path, 'rb') as f:
            metadata = pickle.Unpickler(f).load()
        self.token_list = metadata['token_list']
        self.normal_token_len = metadata['normal_token_len']
        self.total_token_len = len(self.token_list)
        self.special_token_len = self.total_token_len - self.normal_token_len
        self.vector_dim = metadata['vector_dim']

    def make_dict(self):
        """
        トークンからIDを返す辞書を作成
        """
        return {self.token_list[i]:i for i in range(len(self.token_list))}
    
    def get_w(self):
        """
        元となったWと学習したWをひとまとめにしたW全体を返す
        """
        return F.concat([self.W, self.embed_special.W], axis=0)
    
    def set_embed_vector(self, i, v):
        """
        トークンベクトルの情報をid決めうちで書き換える
        Arguments:
            i: 書き換えるトークンのid
            v: 書き換えるベクトル
        """
        if i < self.normal_token_len:
            self.W[i] = v
        else:
            self.embed_special.W.data[i-self.normal_token_len] = v
    
    def __call__(self, xs):
        """
        埋め込みベクトルを返す
        Arguments:
            xs: 埋め込みベクトルを得たいトークンのidの(VariableまたはNdarray)
        """
        if isinstance(xs, chainer.variable.Variable):
            xs = xs.data
        xp = self.xp
        
        # 学習済み(normal), 学習可能(special)それぞれにマスクを作成
        m_normal  = xp.logical_and(0 <= xs, xs < self.normal_token_len)
        m_speical = xp.logical_and(
                self.normal_token_len <= xs, 
                xs < self.total_token_len
            )
        
        # ベクトル取り出し
        y_normal = self.W[xp.where(m_normal, xs, 0)]
        y_special = self.embed_special(xp.where(
                m_speical, 
                xs - self.normal_token_len,
                self.ignore_label
            ).astype('int32'))
        
        # マージして返却
        m_normal = xp.broadcast_to(m_normal[..., None], y_normal.shape)
        m_speical = xp.broadcast_to(m_speical[..., None], y_special.shape)
        return m_speical*y_special + m_normal*y_normal
    
