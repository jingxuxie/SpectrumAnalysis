# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:42:24 2019

@author: HP
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Func_PIGE as fg
import xlsxwriter as xw
import time

time_start=time.time()
folder_name = '6.6.19'
input_path='F://桌面文件2019.2.11//Nuclear Physics Study//DataFiles//'+folder_name
output_file='F://桌面文件2019.2.11//Nuclear Physics Study//PyAnalysis//'+folder_name+'_PIGE.xlsx'
filetype='.csv'

def get_filename(path,filetype):
    name=[]
    files=os.listdir(path)
    for i in files:
        if filetype in i: 
            name.append(i)
    return name

filename=get_filename(input_path,filetype)

HeadLine=7
workbook=xw.Workbook(output_file)
worksheet1=workbook.add_worksheet('ElementAnalysis')#表一输出分析过的数据
worksheet2=workbook.add_worksheet('Raw Results')#表二是原始数据集，以供参考
style_plain=workbook.add_format({'align':'center'})#设置普通居中格式
style_bold=workbook.add_format({'align':'center','bold':True})#设置加粗字体格式
NewFormat=np.array([])
row=1
col=0
for item in filename:
    counter=filename.index(item)+1
    temp=pd.read_csv(input_path+'//'+item,sep=',',header=None,skiprows=HeadLine-1,nrows=1,dtype='str')#读出来字符串，先检查HeadLine最后一行是不是Channel
    if not temp[0][0]=='Channel':#如果不是<<DATA>>，也就意味着出错了
        print(counter,'HeadLine error, '+item+' skipped')
        continue
    temp=pd.read_csv(input_path+'//'+item,sep=',',header=None,skiprows=HeadLine,nrows=32768)#读取文件，以空格分隔，没有标题，调过前面的行，读取4096行
    temp=temp[[1,2]]#取第2、3两列
    Spectrum0=temp.values#dataframe转成array
    Spectrum0_copy=Spectrum0.copy()
    worksheet2.write(0,col,item.replace('.csv','_PIGE'))#输出样品名
    worksheet1.write(row,0,item.replace('.csv','_PIGE'))#输出样品名
    temp=fg.PeakAnalysis_and_Export(Spectrum0_copy,1,col,worksheet1,worksheet2,style_plain,style_bold)#输出峰信息
    if len(NewFormat)==0:#把表一中的数据设成大数组，用于高亮显示感兴趣的点
        NewFormat=temp
    else:
        NewFormat=np.vstack((NewFormat,temp))
    row+=1
    col+=5
    time_end=time.time()
    time_used=time_end-time_start
    time_left=time_used/counter*(len(filename)-counter)#动态修正剩余时间
    time_used=round(time_used)
    time_left=round(time_left)
    str_place='{0:<4}{1:<25}{2:>3}{3:^1}{4:<}{5:}'
    print(str_place.format(counter,item,time_used,'/',time_left,'s',sep=''))
fg.ColorTable(NewFormat,workbook,worksheet1)#高亮显示突出的数据
worksheet1.freeze_panes(1, 1)#冻结窗口
workbook.close()