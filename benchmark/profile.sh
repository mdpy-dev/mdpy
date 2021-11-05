#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# file : profile.sh
# created time : 2021/11/04
# author : Zhenyu Wei
# version : 1.0
# contact : zhenyuwei99@gmail.com
# copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University

nsys profile --stats=true --force-overwrite true -o out/$1 python benchmark.py