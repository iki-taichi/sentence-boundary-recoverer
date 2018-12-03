# coding: utf-8


"""
make_dataset.py

Written_by: Taichi Iki (taici.iki@gmail.com)
Created_at: 2018-12-01
Comments:
展開されたlivedoorコーパスから、文境界モデルの学習データ(json)を出力する
"""


import os
import json


class Args(object):
    corpus_dir = 'dataset/text'
    output_path = 'dataset/livedoor_corpus.json'


def main(args):
    # 2階層目のファイルをまとめて処理していく
    blocks = []
    file_count = 0
    for sub_name in os.listdir(args.corpus_dir):
        path = os.path.join(args.corpus_dir, sub_name)
        
        if sub_name.startswith('.'):
            continue
        if not os.path.isdir(path):
            continue

        for sub_sub_name in os.listdir(path):
            file_path = os.path.join(path, sub_sub_name)
            
            if not os.path.isfile(file_path):
                continue
            if sub_sub_name.startswith('.'):
                continue
            if 'LICENSE.txt' in sub_sub_name:
                continue
            
            blocks.extend(get_blocks_from_file(file_path))
            file_count += 1
    
    with open(args.output_path, 'w') as f:
        json.dump(blocks, f)

    print('file count: %d'%(file_count))
    

def get_blocks_from_file(path):
    """
    前処理のルール
      1. 先頭の2行は除く
      2. 空白行が1行以上続いているところでブロックに分ける
      3. ブロックごとにstripする
    """

    with open(path, 'r') as f:
        lines = f.readlines()
    lines = lines[2:]
    
    blocks = []
    buf = []
    def _flush():
        if len(buf) != 0:
            blocks.append(''.join(buf).strip())
            buf.clear()
    
    for l in lines:
        striped = l.strip()
        if len(striped) == 0:
            _flush()
        else:
            buf.append(l)
    _flush()

    return blocks


if __name__ == '__main__':
    main(Args())