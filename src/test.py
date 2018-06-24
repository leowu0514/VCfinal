import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from data_processing import *
from keras.layers import Input, Dense, Dropout, Reshape, Permute, concatenate, Flatten, Activation
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential, load_model
from keras.optimizers import RMSprop, Adagrad
from keras.callbacks import EarlyStopping

# ffmpeg -f rawvideo -vcodec rawvideo -s 176x144 -r 25 -pix_fmt yuv420p -i table_ratio16.qcif -c:v libx264 -preset veryslow -qp 0 table_ratio16.mp4
# ffplay -f rawvideo -video_size 176x144 soccer_after.qcif
PAIR_NUM = 3
print("Loading data")
try:
	data = load_all_yuv("../data/")
except:
	print("no data exist!")
	sys.exit()

yuv_path_list = sorted(glob.glob(os.path.join("../data/", "*.yuv")))

position = np.zeros(len(data)).astype(int)
tmp = 0
for i in range(len(data)):
	position[i] = tmp
	tmp += data[i].shape[0]

# # normalize
print("Normalizing")
for i in range(len(data)):
	for j in range(len(data[i])):
		for k in range(3):
			data[i][j][:,:,k] /= 255

print("Making training data")
X = [np.zeros((10856,144,176,3)) for i in range(PAIR_NUM)]
Y = np.zeros((10856,144,176,3))
count = 0
for i in range(len(data)):
	print("video", i, len(data[i]), "frames")
	# 取前後 pair_num/2 個 frame, so insert frame
	insert_num = int(PAIR_NUM/2)
	for j in range(insert_num):
		data[i] = np.insert(data[i], 0, data[i][0], axis=0)
		data[i] = np.insert(data[i], len(data[i]), data[i][-1], axis=0)

	for j in range(len(data[i])-insert_num*2):
		# print(j)
		# Y = np.append(Y, data[i][j+int(PAIR_NUM/2)].reshape((1,144,176,3)), axis=0)
		Y[count] = data[i][j+int(PAIR_NUM/2)].reshape((1,144,176,3))
		for k in range(PAIR_NUM):
			# X[k] = np.append(X[k], data[i][j+k].reshape((1,144,176,3)), axis=0)
			X[k][count] = data[i][j+k].reshape((1,144,176,3))

		count += 1
		
print("Split train/valid/test")
T = 9696
Y_x_train = [X[i][:T][:,:,:,0].reshape((T,144,176,1)) for i in range(PAIR_NUM)]
Y_y_train = Y[:T][:,:,:,0].reshape((T,144,176,1))
Y_x_test = [X[i][T:][:,:,:,0].reshape((10856-T,144,176,1)) for i in range(PAIR_NUM)]
Y_y_test = Y[T:][:,:,:,0].reshape((10856-T,144,176,1))

Cb_x_train = [X[i][:T][:,:,:,1].reshape((T,144,176,1)) for i in range(PAIR_NUM)]
Cb_y_train = Y[:T][:,:,:,1].reshape((T,144,176,1))
Cb_x_test = [X[i][T:][:,:,:,1].reshape((10856-T,144,176,1)) for i in range(PAIR_NUM)]
Cb_y_test = Y[T:][:,:,:,1].reshape((10856-T,144,176,1))

Cr_x_train = [X[i][:T][:,:,:,2].reshape((T,144,176,1)) for i in range(PAIR_NUM)]
Cr_y_train = Y[:T][:,:,:,2].reshape((T,144,176,1))
Cr_x_test = [X[i][T:][:,:,:,2].reshape((10856-T,144,176,1)) for i in range(PAIR_NUM)]
Cr_y_test = Y[T:][:,:,:,2].reshape((10856-T,144,176,1))

num = -6 # choose video here (-2~-6)
video_Y = [X[i][position[num]:position[num+1]][:,:,:,0].reshape((position[num+1]-position[num],144,176,1)) for i in range(PAIR_NUM)]
video_Cb = [X[i][position[num]:position[num+1]][:,:,:,1].reshape((position[num+1]-position[num],144,176,1)) for i in range(PAIR_NUM)]
video_Cr = [X[i][position[num]:position[num+1]][:,:,:,2].reshape((position[num+1]-position[num],144,176,1)) for i in range(PAIR_NUM)]

modelY = load_model("Y.h5")
modelCb = load_model("Cb.h5")
modelCr = load_model("Cr.h5")

resY = modelY.predict(video_Y)
resCb = modelCb.predict(video_Cb)
resCr = modelCr.predict(video_Cr)

for i in range(len(resY)):
	resY[i][:,:,0] *= 255
	resCb[i][:,:,0] *= 255
	resCr[i][:,:,0] *= 255

res = np.concatenate([resY, resCb, resCr], axis=3)
yuv_arr_to_yuv_file(res, "result.qcif")



