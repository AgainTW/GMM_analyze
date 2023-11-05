import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def to_train_1(img):
	# 根據顏色合成訓練集:[R.G,B]
	data = np.reshape(img,(img.shape[0]*img.shape[1], 3))
	return data

def to_train_2(img):
	# 根據顏色和位置合成訓練集:[x,y,R.G,B]
	data = np.empty((img.shape[0], img.shape[1], 5), np.dtype('uint8'))
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			data[x,y,:] = [x, y, img[x,y,0], img[x,y,1], img[x,y,2]] 
	data = np.reshape(data,(data.shape[0]*data.shape[1], 5))		
	return data 

def to_train_3(img):
	# 根據灰階和位置合成訓練集:[x,y,Bin]
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	data = np.empty((img.shape[0], img.shape[1], 3), np.dtype('uint8'))
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			data[x,y,:] = [x, y, img[x,y]] 
	data = np.reshape(data,(data.shape[0]*data.shape[1], 3))
	return data 		

def to_img(data,s0,s1,nc):
	data = np.reshape(data,(s0,s1))
	img = np.empty((s0, s1, 3), np.dtype('uint8'))
	for x in range(s0):
		for y in range(s1):
			color = (-data[x][y]-1+nc)*int(255/(nc-1))
			img[x][y] = [color,color,color]
	return img

def gmm(st, img, predict, t, nc, ct):
	'''
	img 	: 訓練圖片來源
	predict	: 預測圖片
	t 	: 轉換方式 
	nc 	: components 數
	ct 	: covariance type
	'''
	cov_type = ['full', 'tied', 'diag', 'spherical']
	g = GaussianMixture(n_components=nc, covariance_type=cov_type[ct])

	if( t==1 ): 
		img_t = to_train_1(img)
		img_test = to_train_1(predict)
	elif( t==2 ): 
		img_t = to_train_2(img)
		img_test = to_train_2(predict)
	elif( t==3 ): 
		img_t = to_train_3(img)
		img_test = to_train_3(predict)

	g.fit(img_t)
	img_p = g.predict(img_test)

	img_p_ = to_img(img_p,predict.shape[0],predict.shape[1],nc)
	name = st+'_t'+str(t)+'_c'+str(nc)+'_ct'+str(ct)+'.jpg'
	cv2.imwrite(name, img_p_)

# read image and convert to np type
img_1 = cv2.imread('./hw4/hw4/soccer1.jpg')
img_2 = cv2.imread('./hw4/hw4/soccer2.jpg')
img_3 = np.concatenate([img_1,img_2])

Scenario = 4
## Scenario 0:
if(Scenario==0):
	gmm('high nc', img_1, img_1, t=2, nc=10, ct=0)

## Scenario 1:
if(Scenario==1):
	for i in range(3):
		for j in range(3):
			for k in range(4):
				gmm('S1', img_1, img_1, t=i+1, nc=j+2, ct=k)

## Scenario 2:
if(Scenario==2):
	for i in range(3):
		for j in range(3):
			for k in range(4):
				gmm('S2',img_1, img_2, t=i+1, nc=j+2, ct=k)

## Scenario 3:
if(Scenario==3):
	for i in range(3):
		for j in range(3):
			for k in range(4):
				gmm('S3_1',img_3, img_1, t=i+1, nc=j+2, ct=k)
				gmm('S3_2',img_3, img_2, t=i+1, nc=j+2, ct=k)

## Scenario 4:
if(Scenario==4):
	gmm('S4',img_1, img_1, t=1, nc=2, ct=0)
	gmm('S4',img_1, img_1, t=1, nc=2, ct=1)
	gmm('S4',img_1, img_1, t=1, nc=2, ct=2)
	gmm('S4',img_1, img_1, t=1, nc=2, ct=3)