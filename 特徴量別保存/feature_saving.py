#変更してみたよ～～
import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager

import pandas as pd


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f'{self.name}_train.ftr'
        self.test_path = Path(self.dir) / f'{self.name}_test.ftr'
    
    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self
    
    @abstractmethod
    def create_features(self):
        raise NotImplementedError
    
    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))

    def create_memo(col_name, desc):
        file_path = Feature.dir + '/_features_memo.csv'
        if not os.path.isfile(file_path):
            with open(file_path,"w"):pass
        with open(file_path, 'r+') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        # 書き込もうとしている特徴量がすでに書き込まれていないかチェック
            col = [line for line in lines if line.split(',')[0] == col_name]
            if len(col) != 0:return
            writer = csv.writer(f)
            writer.writerow([col_name, desc])
            