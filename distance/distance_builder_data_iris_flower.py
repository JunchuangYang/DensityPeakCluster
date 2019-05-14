#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# data reference : R. A. Fisher (1936). "The use of multiple measurements
# in taxonomic problems"

from distance_builder import *
from distance import *


if __name__ == '__main__':
    builder = DistanceBuilder()
    #加载数据，变成array格式
    builder.load_points(r'../data/data_others/iris.data')
    #计算各个数据点之间的距离，并存入文件中
    #SqrtDistance()：距离计算函数名，欧式距离
    builder.build_distance_file_for_cluster(SqrtDistance(), r'../data/data_others/iris_distance.dat')
