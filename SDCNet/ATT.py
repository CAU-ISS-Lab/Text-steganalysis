# encoding=gb18030
import matplotlib.pyplot as plt
from pylab import *                                 #֧������
mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['5', '10', '15', '20', '25']
x = range(len(names))
y = [0.855, 0.84, 0.835, 0.815, 0.81]
y1=[0.86,0.85,0.853,0.849,0.83]
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # �޶�����ķ�Χ
#pl.ylim(-1, 110)  # �޶�����ķ�Χ
plt.plot(x, y, marker='o', mec='r', mfc='w',label=u'y=x^2����ͼ')
plt.plot(x, y1, marker='*', ms=10,label=u'y=x^3����ͼ')
plt.legend()  # ��ͼ����Ч
plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"time(s)�ھ�") #X���ǩ
plt.ylabel("RMSE") #Y���ǩ
plt.title("A simple plot") #����

plt.show()