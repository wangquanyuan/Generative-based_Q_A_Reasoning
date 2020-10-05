import numpy as np
import pandas as pd
import re
from jieba import posseg
import jieba
from tokenizer import segment
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


REMOVE_WORDS = ['|', '[', ']', '语音', '图片', ' ']


def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines


def remove_words(words_list):
    words_list = [word for word in words_list if word not in REMOVE_WORDS]
    return words_list


def parse_data(train_path, test_path):
    train_df = pd.read_csv(train_path, encoding='utf-8')
    #根据“Report”列过滤出缺失数据的行
    train_df.dropna(subset=['Report'], how='any', inplace=True)
    train_df.fillna('', inplace=True)
    #将Question和Dialogue这两列拼接成一列，保存到train_x中
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth',200)
    #print(train_x[45:55])
    print('train_x is ', len(train_x))
    train_x = train_x.apply(preprocess_sentence)
    print('train_x is ', len(train_x))
    train_y = train_df.Report
    print('train_y is ', len(train_y))
    train_y = train_y.apply(preprocess_sentence)
    print('train_y is ', len(train_y))

    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df.fillna('', inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_x = test_x.apply(preprocess_sentence)
    print('test_x is ', len(test_x))
    # 保存时侯要注意索引和表头都去掉，index=None, header=False
    train_x.to_csv('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)
    train_y.to_csv('{}/datasets/train_set.seg_y.txt'.format(BASE_DIR), index=None, header=False)
    test_x.to_csv('{}/datasets/test_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)


def preprocess_sentence(sentence):
    seg_list = segment(sentence.strip(), cut_type='word')
    seg_list = remove_words(seg_list)
    seg_line = ' '.join(seg_list)
    return seg_line


if __name__ == '__main__':
    parse_data('{}/datasets/AutoMaster_TrainSet.csv'.format(BASE_DIR),
               '{}/datasets/AutoMaster_TestSet.csv'.format(BASE_DIR))


