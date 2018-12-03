# sentence-boundary-recoverer
A Japanese sentence boundary recovering model

日本語文字列の境界(句読点や！？)を復元するツール

音声認識結果など境界が失われてしまった文が少し見やすくなります。


## 説明

## 環境

chainer, mojimojiに依存します。

chainer 4.2.0での動作を確認しています。

## 使い方

1. リポジトリをcloneします。

2. リポジトリのルートにいるとして、pythonのインタプリタで使う場合。

```:python
>>> from src.sbr import BoundaryRecoverer
>>> br = BoundaryRecoverer()
>>> br('へえそうなんだ知らなかったよそうでしょ')
'へえそうなんだ知らなかったよ。そうでしょ。'
>>> br('学習データの加工が変でした重複もだいたいなくなりました')
'学習データの加工が変でした。重複もだいたいなくなりました!'
```

## モデルについて(仕組み、評価など)

qiitaの記事をみてください。

## 学習に使用したコーパス

モデルのパラメータ学習において、下記のコーパスを利用しました。

- [livedoor ニュースコーパス](https://www.rondhuit.com/download.html)
