# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:09:43 2019

@author: HP
"""

import multiprocessing
from multiprocessing import Manager

import numpy as np
import time
import logging
import xlsxwriter as xw
import os

def temp_test(i):
    j=i
    return 0

def worker(procnum,return_dict):
    k=0
    for i in range(10000):
        for j in range(1000):
            if i%2==0 and j%2==0:
                k+=1
            if i%2==1 and j%2==1:
                k-=1
    
    temp=temp_test(5)
    return_dict[procnum]=temp
    logging.basicConfig(level=logging.DEBUG, filename='test.log',filemode='a')
    logging.info('information')
    #return_dict[procnum] = procnum


if __name__ == '__main__':

    ts = time.time()
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,return_dict))
        jobs.append(p)
        p.start()
        print(i)

    for proc in jobs:
        proc.join()
    print (return_dict.values())
    te = time.time()
    print("using time: "+str(te - ts)+"s")