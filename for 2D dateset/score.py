import cv2
import pandas as pd
import numpy as np

def score(img_g, loc):
	img_p = cv2.imread(loc , cv2.IMREAD_GRAYSCALE)
	img_g = np.reshape(img_g,(img_g.shape[0]*img_g.shape[1]))
	img_p = np.reshape(img_p,(img_p.shape[0]*img_p.shape[1]))

	count = 0
	for i in range(img_g.shape[0]):
		if(img_g[i]==img_p[i]): count = count + 1
	return count


# 評分
img_1_gray = cv2.imread('./hw4/hw4/soccer2_mask.png' , cv2.IMREAD_GRAYSCALE)

st = './S3/S3_2'
name = []
sc = []
for i in range(3):
	for j in range(3):
		for k in range(4):
			loc = st+'_t'+str(i+1)+'_c'+str(j+2)+'_ct'+str(k)+'.jpg'
			sc.append(score(img_1_gray, loc)/(img_1_gray.shape[0]*img_1_gray.shape[1]))
			name.append(loc)

sc_table = pd.DataFrame()
sc_table["name"] = name
sc_table["score"] = sc
sc_table = sc_table.sort_values(by="score")
sc_table.to_csv('S3_2_score.csv', index=False)
