import re
import numpy as np
import matplotlib.pyplot as plt
from numpy import *

#求和
def fisher_sum(f_list):
	f_sum=[]
	for i in range(len(f_list[0])):
		ssum=0
		for j in range(len(f_list)):
			ssum+=f_list[j][i]
		f_sum.append(ssum)
	return f_sum

#计算平均值
def fisher_mean(f_len,f_sum):
	f_mean=[]
	for num in f_sum:
		f_mean.append(num/f_len)	
	return f_mean

#计算组间平均距离
def fisher_d(pos_mean,neg_mean):
	if len(pos_mean)==len(neg_mean):
		f_d=[]
		for i in range(len(pos_mean)):
			f_d.append((pos_mean[i]-neg_mean[i]))
		return f_d
	else:
		return 0

#类内方差和辅助函数
def fisher_mut_sum(f_list):
	f_mut_sum=[]
	for i in range(len(f_list[0])):
		mut_sum_j=[]
		for j in range(len(f_list[0])):
			mut_sum_k=[]
			for k in range(len(f_list)):
				mut_sum_k.append((f_list[k][i]*f_list[k][j]))
			mut_sum_j.append(sum(mut_sum_k))
		f_mut_sum.append(mut_sum_j)
	return f_mut_sum

#类内方差和
def fisher_s(pos_len,neg_len,pos_mut_sum,neg_mut_sum,pos_sum,neg_sum):
	free_degree=pos_len-1+neg_len-1
	f_s=[]
	for i in range(len(pos_sum)):
		f_s_ls=[]
		for j in range(len(neg_sum)):
			num=(pos_mut_sum[i][j]-1/pos_len*pos_sum[i]*pos_sum[j]
				+neg_mut_sum[i][j]-1/neg_len*neg_sum[i]*neg_sum[j])
			f_s_ls.append(num/free_degree)
		f_s.append(f_s_ls)
	return f_s

#计算期望
def fisher_y(ci,f_mean):
	f_y=0
	for i in range(len(f_mean)):
		f_y+=ci[i]*f_mean[i]
	return f_y

#计算判别值C
def fisher_c(pos_len,neg_len,pos_y,neg_y):
	c=(pos_len*pos_y+neg_len*neg_y)/(pos_len+neg_len)
	return c

def fisher_print_disc(ci,c):
	discstr='y='
	for i in range(len(ci)):
		discstr+=str(round(ci[i],4))+"x"+str(i+1)+"+"
	discstr=discstr.rstrip('+')
	print(discstr,end=' ')
	print("c="+str(round(c,4)))

def fisher_disc(c,ci,uk_list):
	uk_y=c
	for i in range(len(ci)):
		uk_y+=ci[i]*uk_list[i]
	return uk_y

def fisher(pos_list,neg_list,uk_list=[]):
	pos_len=len(pos_list)
	neg_len=len(neg_list)
	pos_sum=fisher_sum(pos_list)
	neg_sum=fisher_sum(neg_list)
	pos_mean=fisher_mean(pos_len,pos_sum)
	neg_mean=fisher_mean(neg_len,neg_sum)
	pos_mut_sum=fisher_mut_sum(pos_list)
	neg_mut_sum=fisher_mut_sum(neg_list)
	d=fisher_d(pos_mean,neg_mean)
	s=fisher_s(pos_len,neg_len,pos_mut_sum,neg_mut_sum,pos_sum,neg_sum)
	ci=np.linalg.solve(s,d)
	pos_y=fisher_y(ci,pos_mean)
	neg_y=fisher_y(ci,neg_mean)
	c=fisher_c(pos_len,neg_len,pos_y,neg_y)
	
	if uk_list:
		if len(uk_list)!=len(ci):
			print("unknown data error!")
		else:
			if pos_y>neg_y:
				if fisher_disc(c,ci,uk_list)>c:
					print('positive')
				else:
					print('negative')
			elif pos_y<neg_y:
				if fisher_disc(c,ci,uk_list)>c:
					print('positive')
				else:
					print('negative')

	return [ci,c]

#模拟训练集
pos_list=[[0,6],[0,8],[2,3],[2,5],[2,8],[3,4],[4,4],[4,6],[5,8]]
neg_list=[[4,2],[5,0],[6,0],[6,2],[7,1],[7,4],[8,0],[8,2],[8,4]]
ci,c=fisher(pos_list,neg_list)
k=ci[0]/ci[1]
#ci=[-1.5,0.4]

#绘图设置
plt.axis('scaled')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(xmax=9,xmin=-5)
plt.ylim(ymax=9,ymin=-5)
pos_color='darkorange'
neg_color='slateblue'

#绘制训练集样本点
pos_x=[point[0] for point in pos_list]
pos_y=[point[1] for point in pos_list]
neg_x=[point[0] for point in neg_list]
neg_y=[point[1] for point in neg_list]
plt.scatter(pos_x,pos_y,s=120,c=pos_color,marker='o',label='pos')
plt.scatter(neg_x,neg_y,s=120,c=neg_color,marker='s',label='neg')

#绘制投影点
for point in pos_list:
	b=point[0]/k+point[1]
	proj_x=(k*b)/(k*k+1)
	proj_y=k*proj_x
	plt.scatter(proj_x,proj_y,s=120,c=pos_color,alpha=0.6,marker='o')
	plt.plot([point[0],proj_x],[point[1],proj_y],linestyle=':',color=pos_color,linewidth=3,alpha=0.5)

for point in neg_list:
	b=point[0]/k+point[1]
	proj_x=(k*b)/(k*k+1)
	proj_y=k*proj_x
	plt.scatter(proj_x,proj_y,s=120,c=neg_color,alpha=0.6,marker='s')
	plt.plot([point[0],proj_x],[point[1],proj_y],linestyle=':',color=neg_color,linewidth=3,alpha=0.5)

#绘制判别线
x=np.linspace(-9,9,100)
y=k*x
plt.plot(x,y,'-r',linewidth=3,color='maroon',alpha=0.6)
plt.text(-4,-3,"y=-1.50x1+0.40x2",weight="bold")
plt.show()

#模型评估--留一法
for point in pos_list:
	pos_sub=pos_list[:]
	pos_sub.remove(point)
	fisher(pos_sub,neg_list,point)

for point in neg_list:
	neg_sub=neg_list[:]
	neg_sub.remove(point)
	fisher(pos_list,neg_sub,point)

#训练集
with open('Fisher-positive.txt') as pos_obj:
	pos_data=pos_obj.read()
	pos_data=re.split('[\t\n]',pos_data)
	pos_data=list(map(float,pos_data))

with open('Fisher-negative.txt') as neg_obj:
	neg_data=neg_obj.read()
	neg_data=re.split('[\t\n ]',neg_data)
	neg_data=list(map(float,neg_data))

pos_list=[pos_data[10*i:10*(i+1)] for i in range(int(len(pos_data)/10))]
neg_list=[neg_data[10*i:10*(i+1)] for i in range(int(len(neg_data)/10))]

fisher(pos_list,neg_list,[0.018,0.0495,-0.3468,-0.036,0.0766,-0.2568,-0.0856,0.0811,-0.3198,-0.0826])
fisher(pos_list,neg_list,[-0.0082,0.0082,-0.1717,0.0082,-0.0136,-0.2425,0.1172,-0.0136,-0.1989,0.0175])