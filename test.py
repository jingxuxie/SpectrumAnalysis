# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:09:01 2019

@author: HP
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter as xw
import xlrd
#import Func_PIXE as fx
from astropy.modeling.models import Lorentz1D
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import time
import multiprocessing 
import concurrent.futures
import cv2

img=cv2.imread("F://test.png",1)
cv2.imshow("1",img)
cv2.waitKey(5000)
cv2.destroyAllWindows()



'''
 
def aa():
 print(np.random.random(1))
 
 
pool = multiprocessing.Pool(3)
for i in range(3):
 pool.apply_async(aa)
 
pool.close()
pool.join()
'''
'''
start=time.time()
#cores = multiprocessing.cpu_count()-7
#pool = multiprocessing.Pool(processes=cores)

def test_for_multiprocessing(l):
    k=0
    for i in range(10**3):
        for j in range(10**3):
            if i%2==0 and j%2==0:
                k+=1
    print(l)
    return

x=range(10**5)
with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(test_for_multiprocessing,x)

time1=time.time()
#x=range(10**1)
#pool.map(test_for_multiprocessing,x)
#for i in x:
#   k=test_for_multiprocessing(1)
time2=time.time()

T1=round(time1-start,3)
T2=round(time2-time1,3)
print(T1,T2)
'''

'''
start=time.time()#测试拷贝速度
test=Spectrum0.copy()
end=time.time()
T=round(end-start,3)
print(T)
'''
'''
#高斯滤波算法，原文可参见https://docs.astropy.org/en/stable/convolution/kernels.html
#几乎跟我的一样好，但是比我还差一点点，哈哈哈哈哈
lorentz = Lorentz1D(1, 0, 1)
x = np.linspace(-5, 5, 100)
data_1D = lorentz(x) + 0.1 * (np.random.rand(100) - 0.5)
gauss_kernel=Gaussian1DKernel(2)
smoothed_data_gauss = convolve(Spectrum0_copy[:,1], gauss_kernel)
box_kernel=Box1DKernel(5)
smoothed_data_box=convolve(Spectrum0_copy[:,1],box_kernel)
plt.plot(smoothed_data_box[5:len(Spectrum0_copy)])
plt.plot(Spectrum0_copy[5:len(Spectrum0_copy),1],color='red')
plt.plot(smoothed_data_gauss[5:len(Spectrum0_copy)],color='blue')
SmoothCurve=fx.SmoothPeak(Spectrum0_copy)
plt.plot(SmoothCurve[:,1],color='green',linewidth=3)
'''
'''
for i in range(4):#说明for无法在里面修改循环变量
    i+=5
    print(i)
'''
'''
ans=np.array([[1,2]])#注意赋值时最好要加.copy()，否则会改变原值
ans1=ans
ans1[0,1]=3
print(ans)
'''
'''
workbook=xw.Workbook('test.xlsx')#测试使用xlsxwriter
worksheet=workbook.add_worksheet('sheet1')
worksheet.write('A1','Hello')
workbook.close()
'''
'''
workbook=xlrd.open_workbook(filename='F://桌面文件2019.2.11//Nuclear Physics Study//Programs//PIXE//PIXEPeaksIdentification.xlsx')
worksheet=workbook.sheet_by_index(0)
ans=worksheet.values()
for i in range(worksheet.nrows):
    print(worksheet.row_values(i))
'''
'''
Element=[]
for i in range(5):
    Element.append('')
'''


