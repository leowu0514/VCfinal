import os
import numpy as np
# from data_processing import *
# from keras.layers import Input, Dense, Dropout, Reshape, Permute, concatenate, Flatten, Activation
# from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, UpSampling2D
# from keras.models import Model, Sequential, load_model
# from keras.optimizers import RMSprop, Adagrad
# from keras.callbacks import EarlyStopping
# from keras.utils import plot_model

def model(pair):
	inp = [[] for i in range(pair)]
	for i in range(pair):
		inp[i] = Input((144, 176, 3))

		# CNNencoder
		x = Conv2D(16, (3,3), activation='relu', padding='same')(inp[i]) # 144 x 176 x 16
		x = MaxPooling2D((2,2))(x) # 72 x 88 x 16
		x = Conv2D(16, (3,3), activation='relu', padding='same')(x) # 72 x 88 x 16
		x = MaxPooling2D((2,2))(x) # 36 x 44 x 16
		x = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 36 x 44 x 8
		x = MaxPooling2D((2,2))(x) # 18 x 22 x 8
		x = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 18 x 22 x 8
		x = MaxPooling2D((2,2))(x) # 9 x 11 x 8
		x = Conv2D(4, (3,3), activation='relu', padding='same')(x) # 9 x 11 x 4
		inp[i] = Flatten()(x)

	# DNN
	flat = concatenate(inp, axis=-1) # 9 x 11 x 4 x pair (396 x pair)
	flat = Dense(9*11*4)(flat)
	flat = Dense(9*11)(flat)
	flat = Dense(9*11*4)(flat)

	# CNNdecoder
	x = Conv2D(4, (3,3), activation='relu', padding='same')(flat) # 9 x 11 x 4
	x = UpSampling2D((2,2))(x) # 18 x 22 x 4
	x = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 18 x 22 x 8
	x = UpSampling2D((2,2))(x) # 36 x 44 x 8
	x = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 36 x 44 x 8
	x = UpSampling2D((2,2))(x) # 72 x 88 x 8
	x = Conv2D(16, (3,3), activation='relu', padding='same')(x) # 72 x 88 x 16
	x = UpSampling2D((2,2))(x) # 144 x 176 x 16
	out = Conv2D(1, (3, 3), activation='sigmoid')(x) # 144 x 176 x 1

	autoencoder = Model(inp, out)
	autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
	return autoencoder

PAIR_NUM = 5
# print("Loading data")
# data = load_all_yuv("../data/")

# # normalize
# print("Normalizing")
# mean = [ [ [[],[],[]] for j in range(len(data[i])) ] for i in range(len(data)) ]
# var = [ [ [[],[],[]] for j in range(len(data[i])) ] for i in range(len(data)) ]
# for i in range(len(data)):
# 	for j in range(len(data[i])):
# 		for k in range(3):
# 			mean[i][j][k] = np.mean(data[i][j][:,:,k])
# 			var[i][j][k] = np.std(data[i][j][:,:,k]+1e-7)
# 			data[i][j][k] = (data[i][j][k] - np.mean(data[i][j][:,:,k]))/np.std(data[i][j][:,:,k]+1e-7)

print("Making training data")
X = [np.zeros((1,144,176,3)) for i in range(PAIR_NUM)]
Y = np.zeros((1,144,176,3))
for i in range(len(data)):
	# 取前後 pair_num/2 個 frame, so insert frame
	insert_num = int(PAIR_NUM/2)
	for j in range(insert_num):
		np.insert(data[i], 0, data[i][0], axis=0)
		np.insert(data[i], len(data[i]), data[i][-1], axis=0)

	for j in range(len(data[i])-insert_num*2):
		Y = np.append(Y, data[i][j+int(PAIR_NUM/2)].reshape((1,144,176,3)), axis=0)
		for k in range(PAIR_NUM):
			X[k] = np.append(X[k], data[i][j+k].reshape((1,144,176,3)), axis=0)
		



