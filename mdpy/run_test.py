#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : run_test.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os, argparse
cur_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
test_dir = os.path.join(cur_dir, 'test')

parser = argparse.ArgumentParser(description='Input of test')
parser.add_argument('-n', type=int, default = 1)
args = parser.parse_args()

if __name__ == '__main__':
    pytest.main(['-r P', '-n %d' %args.n, test_dir])