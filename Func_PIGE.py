# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:47:15 2019

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter as xw
import pandas as pd


def SmoothPeak(Spectrum0):
    Spectrum0_copy=Spectrum0.copy()#copy一下防止修改原变量
    Spectrum=Spectrum0_copy[:,1]#只把第2列取出来，Spectrum是一维的
    n=2
    for k in range(5):
        Length=len(Spectrum)
        aver_temp=np.zeros(Length)
        for i in range(n-1,Length-(n-1)):
            sum_temp=Spectrum[i]
            sum_temp+=sum(Spectrum[i-(n-1):i+(n-1)+1])
            aver_temp[i]=sum_temp/2/n
        Spectrum=0#先置零
        Spectrum=aver_temp[n-1:Length-(n-1)]#赋值
    SmoothCurve=Spectrum0_copy[(k+1)*(n-1):Length+(k+1)*(n-1)-2*(n-1),0]#取出位置向量
    SmoothCurve=np.vstack((SmoothCurve,Spectrum))#水平叠加，变为两行
    SmoothCurve=np.transpose(SmoothCurve)#转置
    if len(SmoothCurve)<3:#如果返回了空值或者背景个数太少的话，防止报错
        SmoothCurve=np.array([[1,0],[2,0],[3,0]])
    #plt.plot(SmoothCurve[:,0],SmoothCurve[:,1],linewidth=5) 
    #plt.plot(Spectrum0_copy[:,0],Spectrum0_copy[:,1])
    return SmoothCurve       
 

def FindBackgroundPeak(x,y):#元素索引是否真正正确还有待进一步检验
    for t in range(10):
        n0=len(x)-1#n0设置成最后一个点的索引
        u=2
        cut=500#cut前后采用不同的方式寻点
        for i in range(u-1,n0):
            sum_temp=sum(y[n0-i:n0-i+u])
            aver_temp=sum_temp/u
            if x[n0-i]>cut:
                if y[n0-i]-y[n0-i+1]>aver_temp/3 and y[n0-i]-y[n0-i+1]>0.5:
                    x=np.delete(x,n0-i,axis=0)#删除行
                    y=np.delete(y,n0-i,axis=0)
            if x[n0-i]<=cut:
                if n0-i>0:
                    if y[n0-i]>y[n0-i+1] and y[n0-i]>y[n0-i-1]:
                        x=np.delete(x,n0-i,axis=0)#删除行
                        y=np.delete(y,n0-i,axis=0)
    if len(x)==0 or len(y)==0:#如果没有，就返回0,这是防止在极端情况下报错
        x=0
        y=0
    return x,y


def ShiftSpectrum(Spectrum0):
    Spectrum0_copy=Spectrum0.copy()#先复制一份
    shift=20#设定位移量
    if len(Spectrum0_copy)==0:#为了防止报错，加一个判断条件
        Spectrum0_copy=np.array([[0,0]])#设为一行两列的0
    Spectrum0_copy[:,0]=Spectrum0_copy[:,0]+shift+1#将横坐标平移n+1,+1是为了防止重叠
    Spectrum=np.array([[i] for i in range(1,shift+1)])#产生1到n的一列向量
    temp=np.array([[0] for i in range(1,shift+1)])#产生n的零的列向量
    Spectrum=np.hstack((Spectrum,temp))#合并为两列
    Spectrum=np.vstack((Spectrum,Spectrum0_copy))#与之前移动的谱合并，填上中间的
    last_temp=Spectrum[len(Spectrum)-1,0]#找到最后一个的横坐标
    temp1=np.array([[last_temp+i] for i in range(1,shift+1)])#生成后面补的横坐标
    temp=np.hstack((temp1,temp))#把后面的合并为两列
    Spectrum=np.vstack((Spectrum,temp))#合并后面添加的0
    shift=shift+1#因为其实向后挪动了shift+1个，在平移横坐标的时候
    return Spectrum,shift


def SmoothBackground(Spectrum0):
    Spectrum0_copy=Spectrum0.copy()#先复制一份
    Cut=500
    CutPos=0#先置零以防出错
    for CutPos in range(len(Spectrum0_copy)):
        if Spectrum0_copy[CutPos,0]>Cut:
            break
    Spectrum=0
    Spectrum=Spectrum0_copy[CutPos:,]#取CutPos之后的值
    Spectrum,shift=ShiftSpectrum(Spectrum)#移位
    Spectrum_copy=Spectrum.copy()#复制一份，用于记录其横坐标
    Spectrum=Spectrum[:,1]#只取其第2列，将Spectrum改为1维
    n=10
    for k in range(10):
        Length=len(Spectrum)
        aver_temp=np.zeros(Length)
        for i in range(n-1,Length-(n-1)):
            sum_temp=Spectrum[i]
            sum_temp+=sum(Spectrum[i-(n-1):i+(n-1)+1])
            aver_temp[i]=sum_temp/2/n
        Spectrum=0#先置零
        Spectrum=aver_temp[n-1:Length-(n-1)]#赋值
    SmoothBackground=Spectrum_copy[(k+1)*(n-1):Length+(k+1)*(n-1)-2*(n-1),0]#取出位置向量
    SmoothBackground=np.vstack((SmoothBackground,Spectrum))#水平叠加，变为两行
    SmoothBackground=np.transpose(SmoothBackground)#转置
    SmoothBackground[:,0]=SmoothBackground[:,0]-shift#还原平移的坐标
    for i in range(len(SmoothBackground)):#搜索截断点
        if SmoothBackground[i,0]>Cut:
            break
    SmoothBackground=SmoothBackground[i:len(SmoothBackground)-1,:]#只取截断点之后的值
    SmoothBackground=np.vstack((Spectrum0_copy[range(CutPos),:],SmoothBackground))#补上截断点之前的,后面的点就不管了
    if len(SmoothBackground)<3:#如果返回了空值或者背景个数太少的话，防止报错
        SmoothBackground=np.array([[1,0],[2,0],[3,0]])
    #plt.plot(Spectrum0_copy[:,0],Spectrum0_copy[:,1],color='green')
    #plt.plot(SmoothBackground[:,0],SmoothBackground[:,1],color='blue')
    return SmoothBackground


def findpeaks(data):#如果输入数据小于3或没有极大值都会返回空值
    #输入为list或array，输出为array列向量
    peaks=np.array([])
    pos=np.array([])
    Length=len(data)
    if len(data)<3:
        return peaks,pos
    for i in range(1,Length-1):
        if data[i-1]<data[i]>=data[i+1]:#根据MATLAB模式这样设置
            peaks=np.hstack((peaks,data[i]))
            pos=np.hstack((pos,i))#向后叠加
            if len(pos)==1:#一开始置空默认是float，位置需要转为整型
                pos=int(pos)
                pos=np.array(pos)#防止pos仅有一个的时候被认成int而报错
    peaks=peaks.reshape((-1,1))#转置成列向量
    pos=pos.reshape((-1,1))
    if len(peaks)==0:#防止报错
        peaks=np.array([[0]])
        pos=np.array([[0]])
    return peaks,pos


def FindBackground(Spectrum0):
    Spectrum0_copy=Spectrum0.copy()#先复制
    SmoothSpectrum=SmoothPeak(Spectrum0_copy)#光滑
    a,b=findpeaks(-SmoothSpectrum[:,1])#寻峰
    x=SmoothSpectrum[b,0]#取峰位置
    y=-a#取峰高
    x,y=FindBackgroundPeak(x,y)
    Peaks=np.hstack((x,y))
    Background=SmoothBackground(Peaks)#已经在SmoothBackground中设置不会返回空值
    #plt.plot(Spectrum0_copy[:,0],Spectrum0_copy[:,1],color='green')
    #plt.plot(Background[:,0],Background[:,1],color='blue')
    return Background


def CutBackground(Spectrum0):
    #返回列向量CutOffBackground和列向量Background
    Spectrum0_copy=Spectrum0.copy()#先复制
    Spectrum=SmoothPeak(Spectrum0_copy)
    Background=FindBackground(Spectrum0_copy)#已经在FindBackground中设置不会返回空值
    DiffSBStart=0#记录有背景的谱的起点
    k=0#记录前一个点有没有对应点，k=0则没有，用于求起点DiffSBStart
    DiffSB=np.array([])#先定义空值，防止出错
    j=0
    for i in range(len(Spectrum)):
        while j<len(Background)-1:#前面已设置Background最少有3行,j必不为空
            if Background[j,0]<=Spectrum[i,0]<=Background[j+1,0]:#找背景对应点
                k=1#找到对应的背景点              
                break
            j+=1
        if j==len(Background)-1 and k==0:#如果没找到对应点，且前一个也没有对应点,则起点+1
            DiffSBStart+=1
            k=0#k重新置零
            j=-1#j也置零重新搜索
            continue
        if j==len(Background)-1 and k==1:#如果到了后面没有背景了，就退出
            break
        d=Background[j+1,0]-Background[j,0]#插值计算当地背景值
        d1=Spectrum[i,0]-Background[j,0]
        r=d1/d
        temp=Background[j,1]+r*(Background[j+1,1]-Background[j,1])
        DiffSB=np.hstack((DiffSB,temp))#依次往后附加新值
    CutOffBackground=Spectrum[DiffSBStart:len(DiffSB)+DiffSBStart,1]-DiffSB
    x_temp=Spectrum[DiffSBStart:len(DiffSB)+DiffSBStart,0]
    CutOffBackground=CutOffBackground.reshape((-1,1))#转置成列向量
    x_temp=x_temp.reshape((-1,1))
    CutOffBackground=np.hstack((x_temp,CutOffBackground))#叠加成两列的向量
    Background=DiffSB
    Background=Background.reshape((-1,1))#变成列向量
    if len(CutOffBackground)==0 or len(Background)==0:#如果是空的话
        CutOffBackground=np.array([[0,0],[0,0],[0,0]])#置为三行0
        Background=np.array([[0,0,0]])
    #plt.plot(Spectrum0_copy[:,0],Spectrum0_copy[:,1],color='green')
    #plt.plot(CutOffBackground[:,0],CutOffBackground[:,1],color='green')
    return CutOffBackground,Background
    

def FindPeak(Spectrum0):
    Spectrum0_copy=Spectrum0.copy()
    CutBackgroundSpectrum,Background=CutBackground(Spectrum0_copy)
    CutBackgroundSpectrum_copy=CutBackgroundSpectrum.copy()
    Background_copy=Background.copy()
    Spectrum=CutBackgroundSpectrum.copy()#必须用copy进行处理，否则后面会改变CutBkS
    Spectrum[:,[1]]=Spectrum[:,[1]]+Background#取索引时写成[1]出来就是一列的向量
    a,b=findpeaks(Spectrum[:,[1]])
    x=Spectrum[b,0]#x和y均为一列的向量
    y=a
    z=y.copy()#z用于记录需要消去的点
    Background=Background[b[:,0]]#返回结果是一列的向量
    Background=np.hstack((x,Background))#叠加上位置信息，变为两列
    diff=np.zeros((len(z),1))#生成一列0向量
    for i in range(len(x)):
        diff[i]=y[i]-abs(Background[i,1])#diff为峰高
        if x[i]<150 and diff[i]<20:#标记要清除指定条件下的点
            z[i]=-1
        elif 150<=x[i]<300 and diff[i]<12:
            z[i]=-1
        elif 300<=x[i]<500 and diff[i]<8:
            z[i]=-1
        elif 500<=x[i]<1000 and diff[i]<6:
            z[i]=-1
        elif 1000<=x[i]<1500 and diff[i]<5:
            z[i]=-1
        elif 1500<=x[i] and diff[i]<3:
            z[i]=-1
        if Background[i,1]>100 and diff[i]<30:
            z[i]=-1
        elif Background[i,1]>50 and diff[i]<25:
            z[i]=-1
        elif Background[i,1]>30 and diff[i]<20:
            z[i]=-1
        elif Background[i,1]>10 and diff[i]<15:
            z[i]=-1
        elif Background[i,1]>5 and diff[i]<8:
            z[i]=-1
        elif Background[i,1]>2 and diff[i]<5:
            z[i]=-1
    j=0#j用于记录真峰的个数
    for i in range((len(x))):#去除被标记的假峰
        if z[i]!=-1:
            x[j]=x[i]
            y[j]=y[i]
            diff[j]=diff[i]
            Background[j,:]=Background[i,:]
            j+=1
    x=x[range(j)]#留下真峰
    z=x.copy()
    y=y[range(j)]
    diff=diff[range(j)]
    Background=Background[range(j),:]
    i=0
    while i<len(x)-1:
        if Background[i,0]<150:
            DiffThreshold=10
        if Background[i,0]>=150:
            DiffThreshold=7.5
        if Background[i,1]>=5:
            DiffThreshold=8
        if Background[i,1]>=20:
            DiffThreshold=10
        if Background[i,1]>=50:
            DiffThreshold=20
        if Background[i,1]>=100:
            DiffThreshold=30
        u=i#u用来记录当前待比较的点
        for k in range(i+1,len(x)):#将第u个峰与第k个峰比较
            for j in range(len(CutBackgroundSpectrum)):#找峰上的对应点
                if CutBackgroundSpectrum[j,0]==x[u]:
                    j0=j
                if CutBackgroundSpectrum[j,0]==x[k]:
                    break
            if min(diff[u],diff[k])-min(CutBackgroundSpectrum[j0:j+1,1])<DiffThreshold and x[k]-x[u]<15:
               if diff[u]<=diff[k]:#小的记做0，大的记做2
                   z[u]=0
                   z[k]=2
                   u=k
               if diff[u]>diff[k]:
                   z[u]=2
                   z[k]=0
            else:#如果两峰不重叠的话
                u=k-1
                break
        if k==len(x)-1:#如果已经比较过了最后一个峰
            break
        i=u+1
    j=0
    for i in range(len(x)):#删掉重叠的峰
        if z[i]!=0:
            x[j]=x[i]
            y[j]=y[i]
            j+=1
    x=x[range(j)]
    y=y[range(j)]
    Peak=np.hstack((x,y))#叠加为两列
    if len(Peak)==0:#如果没有峰的话
        Peak=np.array([[0,0]])
    #plt.scatter(x,y,s=200,marker='*',color='red')
    print('FindPeak')
    return Peak,CutBackgroundSpectrum_copy,Background_copy


def PeakArea(Spectrum0):
    Spectrum0_copy=Spectrum0.copy()
    Peak,CutBackgroundSpectrum,Background=FindPeak(Spectrum0_copy)
    if len(Peak)==1 and Peak[0,0]==0:#如果没有读取到峰值，直接返回四列0
        Peak_and_Area=np.array([[0,0,0,0]])
        return Peak_and_Area
    Background=np.hstack((CutBackgroundSpectrum[:,[0]],Background))#叠加第一列位置
    FWHM=np.zeros([len(Peak[:,0]),2])#半峰宽，左右底峰，都先全置为0
    PeakBase1=np.zeros([len(Peak[:,0]),2])
    PeakBase2=np.zeros([len(Peak[:,0]),2])
    for i in range(len(Peak[:,0])):
        for j in range(len(CutBackgroundSpectrum[:,0])):#先找峰在谱中的对应点
            if CutBackgroundSpectrum[j,0]==Peak[i,0]:#按位置找
                Peak[i,1]=Peak[i,1]-Background[j,1]#峰减去背景值
                break
        threshold=2.5
        threshrate=300
        for k in range(200):#搜索左半峰
            if CutBackgroundSpectrum[j-k,1]<=CutBackgroundSpectrum[j,1]/2:
                FWHM[i,0]=CutBackgroundSpectrum[j,0]-CutBackgroundSpectrum[j-k,0]
                d=2*FWHM[i,0]
                break
            if i>0:#限制左半峰位置不超过前一个右峰底的位置
                if CutBackgroundSpectrum[j-k,0]<=PeakBase2[i-1,0]:#到上一个峰右底端为止
                    FWHM[i,0]=CutBackgroundSpectrum[j,0]-PeakBase2[i-1,0]
                    d=2*FWHM[i,0]
                    break
        for k in range(400):#搜索左峰宽
            if j-k<0:
                PeakBase1[i,:]=CutBackgroundSpectrum[0,:]#防止超出左边界
            else:
                if CutBackgroundSpectrum[j,0]-CutBackgroundSpectrum[j-k,0]>=d\
                and (CutBackgroundSpectrum[j-k,1]<threshold or CutBackgroundSpectrum[j-k,1]<Peak[i,1]/threshrate):
                    PeakBase1[i,:]=CutBackgroundSpectrum[j-k,:]
                    break
                if i>0:#限制左峰不能超过前一个右峰底
                    if CutBackgroundSpectrum[j-k,0]==PeakBase2[i-1,0]:#如果到了上一个峰右底端
                        PeakBase1[i,1]=min(CutBackgroundSpectrum[j-k:j+1,1])#取之间的最小值
                        index=np.where(PeakBase1[i,1]==CutBackgroundSpectrum[j-k:j+1,1])
                        index=np.array(index)[0,0]
                        PeakBase1[i,0]=CutBackgroundSpectrum[j-k+np.array(index),0]
                        break
        if PeakBase1[i,0]==0:#如果还没有确定左峰的话,则取做峰自身，保证程序运行
             PeakBase1[i,:]=CutBackgroundSpectrum[j,:]
        if i<len(Peak[:,0])-1:#找到下一个峰的位置并记录两者中最小值的位置
            for m in range(200):
                min_point=m#先定义，防止报错
                if CutBackgroundSpectrum[j+m,0]==Peak[i+1,0]:
                    min_temp=min(CutBackgroundSpectrum[j:j+m+1,1])
                    min_point=np.where(CutBackgroundSpectrum[j:j+m+1,1]==min_temp)
                    min_point=np.array(min_point,dtype=int)
                    min_point=min_point[0,0]#根据MATLAB程序，选最前面的最小值
                    break
        for k in range(200):#搜索右半峰
            if j+k==len(CutBackgroundSpectrum[:,1])-1:#如果超出右边界
                FWHM[i,1]=CutBackgroundSpectrum[j+k,0]-CutBackgroundSpectrum[j,0]
                d=2*FWHM[i,1]
                break
            if CutBackgroundSpectrum[j+k,1]<=CutBackgroundSpectrum[j,1]/2:#按半峰搜索
                FWHM[i,1]=CutBackgroundSpectrum[j+k,0]-CutBackgroundSpectrum[j,0]
                d=2*FWHM[i,1]
                break
            if i<len(Peak[:0])-1:#右半峰不能超过两峰之间的最低值
                if k>=min_point:
                    FWHM[i,1]=CutBackgroundSpectrum[j+min_point,0]-CutBackgroundSpectrum[j,0]
                    d=2*FWHM[i,1]
                    break
        for k in range(400):#搜索右峰宽
            if j+k==len(CutBackgroundSpectrum[:,1])-1:#如果超出右边界
                PeakBase2[i,:]=CutBackgroundSpectrum[j+k,:]
                break
            if CutBackgroundSpectrum[j+k,0]-CutBackgroundSpectrum[j,0]>=d\
            and (CutBackgroundSpectrum[j+k,1]<threshold or CutBackgroundSpectrum[j+k,1]<Peak[i,1]/threshrate):
                PeakBase2[i,:]=CutBackgroundSpectrum[j+k,:]
                break
            if i<len(Peak[:,0])-1:#右峰限制条件，不能超过两峰之间的最低谷
                if k>=min_point:
                    PeakBase2[i,:]=CutBackgroundSpectrum[j+min_point,:]
                    break
        if PeakBase2[i,0]==0:#如果还没有确定右峰底的话
            PeakBase2[i,:]=CutBackgroundSpectrum[j,:]
    FWHM=FWHM[:,[0]]+FWHM[:,[1]]#将左右半峰宽加起来，这样得到的是一列向量
    Area=np.zeros([len(Peak[:,0]),1])
    j=0
    for i in range(len(Peak[:,0])):#开始计算面积
        sum_temp=0
        j=0
        while j<len(CutBackgroundSpectrum[:,0])-1:
            if CutBackgroundSpectrum[j,0]==PeakBase1[i,0]:#左峰底
                while CutBackgroundSpectrum[j,0]<PeakBase2[i,0]:#右峰底
                    h1=CutBackgroundSpectrum[j,1]#梯形上底
                    h2=CutBackgroundSpectrum[j+1,1]#梯形下底
                    h=CutBackgroundSpectrum[j+1,0]-CutBackgroundSpectrum[j,0]
                    sum_temp+=(h1+h2)*h/2
                    j+=1
                Area[i]=sum_temp
                break
            j+=1
    #plt.scatter(PeakBase1[:,0],PeakBase1[:,1],marker='*',color='blue',s=200)
    #plt.scatter(PeakBase2[:,0],PeakBase2[:,1],marker='*',color='purple',s=200)
    Peak_and_Area=np.array([])
    Peak_and_Area=np.hstack((Peak,Area,FWHM))#合并为4列
    print('PeakArea')
    return Peak_and_Area
    

def NormalizePeakAreaByArgon(Spectrum0):
    Spectrum0_copy=Spectrum0.copy()
    Peak_and_Area=PeakArea(Spectrum0_copy)
    ArgonArea=0
    for i in range(len(Peak_and_Area[:,0])):
        if 770<Peak_and_Area[i,0]<773:
            ArgonArea=Peak_and_Area[i,2]#提取Argon面积
            break
    if ArgonArea==0:
        ArgonArea=1
    Peak_and_AreaNorm=Peak_and_Area.copy()
    Peak_and_AreaNorm[:,2]=Peak_and_AreaNorm[:,2]*100/ArgonArea#按Argon峰面积为100计
    return Peak_and_AreaNorm


def CutDecimal(Data,decimal):
    for i in range(len(Data)):
        Data[i]=round(Data[i],decimal)
    return Data


def PeakAnalysis_and_Export(Spectrum0,row,col,worksheet1,worksheet2,style1,style2):
    Spectrum0_copy=Spectrum0.copy()
    Peak_Area=NormalizePeakAreaByArgon(Spectrum0_copy)#计算的峰位置和面积信息
    Peak_Area[:,0]=CutDecimal(Peak_Area[:,0],1)#峰位置一位小数
    Peak_Area[:,2]=CutDecimal(Peak_Area[:,2],0)#峰面积保留整数
    Peak_Area[:,3]=CutDecimal(Peak_Area[:,3],1)#半峰宽，保留一位小数
    Element,NewFormat=IdentifyPeak(Peak_Area[:,0],Peak_Area[:,2],worksheet1,style2)
    for i in range(len(Peak_Area[:,0])):
        worksheet2.write(row,col,Element[i],style1)
        worksheet2.write(row,col+1,Peak_Area[i,0],style1)
        worksheet2.write(row,col+2,Peak_Area[i,2],style1)
        worksheet2.write(row,col+3,Peak_Area[i,3],style1)
        row+=1
    NewFormat=NewFormat.reshape((1,-1))#变成一行向量
    print('PeakAnalysis&Export')
    return NewFormat


def IdentifyPeak(PeakPos,PeakArea,worksheet1,style):
    #存放峰信息文件的地址
    filepath='F://桌面文件2019.2.11//Nuclear Physics Study//PyCode//PIGEPeaksIdentification.xlsx'
    temp=pd.read_excel(filepath,header=None,sheet_name='Peaks&Order')#读取峰信息
    ElementPeakPos=temp[0].values
    ElementName=temp[1]
    PeakOrder=temp[2].values
    temp=pd.read_excel(filepath,header=None,sheet_name='ElementOrder')#读取元素级别信息
    ElementOrder=temp[0]
    temp=pd.read_excel(filepath,header=None,sheet_name='ElementWant')#读取要在第一行输出的元素信息
    temp=temp.T#原本是行向量，转置成列
    ElementWant2Know=temp[0]
    col=1
    for item in ElementWant2Know:#写入表格中
        worksheet1.write(0,col,item,style)
        col+=1
    Element=[]#先置一个空的list
    OrderNumber=[0]#空的级别序号
    for i in range(len(PeakPos)):
        Element.append('')
        OrderNumber.append(0)
    ErrorRange=0.5#设置误差范围
    for i in range(len(PeakPos)):
        for eo in range(len(ElementOrder)):
            ElementPeakPosM=[]
            ElementNameM=[]
            PeakOrderM=[]
            for en in range(len(ElementName)):
                if ElementOrder[eo]==ElementName[en]:#找到对应的元素，之后再找其谱线
                    ElementPeakPosM.append(ElementPeakPos[en])
                    ElementNameM.append(ElementName[en])
                    PeakOrderM.append(PeakOrder[en])
            for j in range(len(ElementPeakPosM)):
                if ElementPeakPosM[j]-ErrorRange<=PeakPos[i]<=ElementPeakPosM[j]+ErrorRange\
                and PeakOrderM[j]==1 and len(Element[i])==0:
                    #如果在范围内且是1级峰且未被填充过
                    Element[i]=ElementNameM[j]
                    break
    #再另找一遍二级峰
    for i in range(len(PeakPos)):
        for eo in range(len(ElementOrder)):
            ElementPeakPosM=[]
            ElementNameM=[]
            PeakOrderM=[]
            for en in range(len(ElementName)):
                if ElementOrder[eo]==ElementName[en]:#找到对应的元素，之后再找其谱线
                    ElementPeakPosM.append(ElementPeakPos[en])
                    ElementNameM.append(ElementName[en])
                    PeakOrderM.append(PeakOrder[en])
            for j in range(len(ElementPeakPosM)):
                if ElementPeakPosM[j]-ErrorRange<=PeakPos[i]<=ElementPeakPosM[j]+ErrorRange\
                and PeakOrderM[j]==2 and len(Element[i])==0:
                    for k in range(len(Element)):#搜索前面是不是已经有了一级峰
                        if Element[k]==ElementNameM[j]:
                            Element[i]=ElementNameM[j]
                            break
                   
    NewFormat=np.zeros(len(ElementWant2Know))
    for i in range(len(ElementWant2Know)):
        for j in range(len(Element)):
            if ElementWant2Know[i] in Element[j]:
                NewFormat[i]+=PeakArea[j]
    print('IdentifyPeak')
    return Element,NewFormat
                    

def ColorTable(Data,workbook,worksheet1):
    style_plain=workbook.add_format({'align':'center'})#设置普通居中格式
    color_list=['#BBFFFF','#FFDEAD','#76EEC6','#EE6363','#FFD700','#FFB6C1','#C6E2FF']
    style_highlight=[]
    for color in color_list:
        style_highlight.append(workbook.add_format({'align':'center','bg_color':color}))
    for i in range(Data.shape[1]):#对于每一列
        aver_temp=np.mean(Data[:,i])#求平均值
        median_temp=np.median(Data[:,i])
        for j in range(Data.shape[0]):#对于每一行的点
            if (Data[j,i]>aver_temp and Data[j,i]>2*median_temp) and Data[j,i]!=0:#按中值和平均值结合选取高亮点
                style=style_highlight[i%len(color_list)]
            else:
                style=style_plain
            worksheet1.write(j+1,i+1,Data[j,i],style)

#CutOffBackground,Background=CutBackground(Spectrum0)
#Spectrum0=np.array([[1,0],[2,0],[3,0]])
#PeakAnalysis_and_Export(Spectrum0,1,1)
#IdentifyPeak(1,2)