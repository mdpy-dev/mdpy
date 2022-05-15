#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : space_filling_curve.py
created time : 2022/05/13
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np

def morton(x, y, z):
  x = (x | (x << 16)) & 0x030000FF
  x = (x | (x <<  8)) & 0x0300F00F
  x = (x | (x <<  4)) & 0x030C30C3
  x = (x | (x <<  2)) & 0x09249249

  y = (y | (y << 16)) & 0x030000FF
  y = (y | (y <<  8)) & 0x0300F00F
  y = (y | (y <<  4)) & 0x030C30C3
  y = (y | (y <<  2)) & 0x09249249

  z = (z | (z << 16)) & 0x030000FF
  z = (z | (z <<  8)) & 0x0300F00F
  z = (z | (z <<  4)) & 0x030C30C3
  z = (z | (z <<  2)) & 0x09249249

  return x | (y << 1) | (z << 2)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    num_points = 12
    index = np.zeros([num_points**3, 3], np.int32)
    morton_index = np.zeros(num_points**3, np.int32)

    cur_index = 0
    for i in range(num_points):
        for j in range(num_points):
            for k in range(num_points):
                index[cur_index, :] = [i, j, k]
                morton_index[cur_index] = morton(i, j, k)
                cur_index += 1

    sort_index = index[np.argsort(morton_index), :]

    with open('/home/zhenyuwei/nutstore/ZhenyuWei/Note_Research/mdpy/mdpy/mdpy/utils/test.xyz', 'w') as f:
        print('%d' %(num_points**3), file=f)
        print('test' , file=f)
        for i in range(num_points**3):
            print('H %.2f %.2f %.2f %d' %(sort_index[i, 0], sort_index[i, 1], sort_index[i, 2], i), file=f)