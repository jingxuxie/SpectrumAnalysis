# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:27:02 2019

@author: HP
"""

import multiprocessing
import time

start=time.time()
x=range(10)
def test_for_multiprocessing(l):
    k=0
    for i in range(10000):
        for j in range(1000):
            if i%2==0 and j%2==0:
                k+=1
            if i%2==1 and j%2==1:
                k-=1
    return k

def worker(num):
    """Returns the string of interest"""
    return "worker %d" % num
 
def main(x):
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    results = pool.map(test_for_multiprocessing,x)
 
    pool.close()
    pool.join()
 
    #for result in results:
        # prints the result string in the main process
     #   print(result)
if __name__ == '__main__':
    for i in x:
       k=test_for_multiprocessing(1)
    
    time1=time.time()
    time2=time.time()
    
    T1=round(time1-start,3)
    T2=round(time2-time1,3)
    print(T1,T2)















