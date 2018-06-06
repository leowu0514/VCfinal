import os
import numpy as np
from data_processing import *
from keras.layers import Input, Dense, Dropout, Reshape, dot, Permute, average, concatenate, Flatten, Activation
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, UpSampling2D, GRU
from keras.models import Model, Sequential, load_model
from keras.optimizers import RMSprop, Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def model2(): # only CNN
	inp = Input((144, 176, 1))

	# CNNencoder
	x = Conv2D(16, (3,3), activation='relu', padding='same')(inp) # 144 x 176 x 16
	x = MaxPooling2D((2,2), padding='same')(x) # 72 x 88 x 16
	x = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 72 x 88 x 16
	x = MaxPooling2D((2,2), padding='same')(x) # 36 x 44 x 16
	x = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 36 x 44 x 8
	
	# CNNdecoder
	x = UpSampling2D((2,2))(x) # 72 x 88 x 8
	x = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 72 x 88 x 16
	x = UpSampling2D((2,2))(x) # 144 x 176 x 16
	out = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x) # 144 x 176 x 1

	# print(inp)

	autoencoder = Model(inp, out)
	autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
	# autoencoder.compile(optimizer=RMSprop(), loss='mean_squared_error')
	return autoencoder

def model3(pair):
	inp = [Input((144, 176, 1)) for i in range(pair)]
	tmp = [[] for i in range(pair)]
	flat = [[] for i in range(pair)]
	for i in range(pair):
		# CNNencoder
		x = Conv2D(16, (3,3), activation='relu', padding='same')(inp[i]) # 144 x 176 x 16
		x = MaxPooling2D((2,2), padding='same')(x) # 72 x 88 x 16
		x = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 72 x 88 x 8
		x = MaxPooling2D((2,2), padding='same')(x) # 36 x 44 x 8
		tmp[i] = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 36 x 44 x 8
		tmp[i] = Reshape((1,36*44*8))(tmp[i])
		flat[i] = Flatten()(tmp[i])
		# flat[i] = Dense(36*44)(flat[i])
		flat[i] = Dense(1)(flat[i])
		# tmp[i] = Dense(36*44)(tmp[i])
	
	frames = concatenate(tmp, axis=1)
	weight = concatenate(flat, axis=-1)
	# merge = average(tmp)
	print(tmp[0].shape, weight.shape, frames.shape)
	merge = dot([weight, frames], 1)
	merge = Reshape((36,44,8))(merge)

	# CNNdecoder
	x = UpSampling2D((2,2))(merge) # 72 x 88 x 8
	x = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 72 x 88 x 16
	x = UpSampling2D((2,2))(x) # 144 x 176 x 16
	out = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x) # 144 x 176 x 1

	# print(inp)

	autoencoder = Model(inp, out)
	autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
	# autoencoder.compile(optimizer=RMSprop(), loss='mean_squared_error')
	return autoencoder


PAIR_NUM = 3
print("Loading data")
data = load_all_yuv("../data/")

# normalize
print("Normalizing")
# mean = [ [ [[],[],[]] for j in range(len(data[i])) ] for i in range(len(data)) ]
# var = [ [ [[],[],[]] for j in range(len(data[i])) ] for i in range(len(data)) ]
for i in range(len(data)):
	for j in range(len(data[i])):
		for k in range(3):
			data[i][j][:,:,k] /= 255
			# mean[i][j][k] = np.mean(data[i][j][:,:,k])
			# var[i][j][k] = np.std(data[i][j][:,:,k]+1e-7)
			# data[i][j][:,:,k] = (data[i][j][:,:,k] - np.mean(data[i][j][:,:,k]))/np.std(data[i][j][:,:,k]+1e-7)

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
V = 9696
Y_x_train = [X[i][:T][:,:,:,0].reshape((T,144,176,1)) for i in range(PAIR_NUM)]
Y_y_train = Y[:T][:,:,:,0].reshape((T,144,176,1))
Y_x_valid = [X[i][T:V][:,:,:,0].reshape((V-T,144,176,1)) for i in range(PAIR_NUM)]
Y_y_valid = Y[T:V][:,:,:,0].reshape((V-T,144,176,1))
Y_x_test = [X[i][V:][:,:,:,0].reshape((10856-V,144,176,1)) for i in range(PAIR_NUM)]
Y_y_test = Y[V:][:,:,:,0].reshape((10856-V,144,176,1))

Cb_x_train = [X[i][:T][:,:,:,1].reshape((T,144,176,1)) for i in range(PAIR_NUM)]
Cb_y_train = Y[:T][:,:,:,1].reshape((T,144,176,1))
Cb_x_valid = [X[i][T:V][:,:,:,1].reshape((V-T,144,176,1)) for i in range(PAIR_NUM)]
Cb_y_valid = Y[T:V][:,:,:,1].reshape((V-T,144,176,1))
Cb_x_test = [X[i][V:][:,:,:,1].reshape((10856-V,144,176,1)) for i in range(PAIR_NUM)]
Cb_y_test = Y[V:][:,:,:,1].reshape((10856-V,144,176,1))

Cr_x_train = [X[i][:T][:,:,:,2].reshape((T,144,176,1)) for i in range(PAIR_NUM)]
Cr_y_train = Y[:T][:,:,:,2].reshape((T,144,176,1))
Cr_x_valid = [X[i][T:V][:,:,:,2].reshape((V-T,144,176,1)) for i in range(PAIR_NUM)]
Cr_y_valid = Y[T:V][:,:,:,2].reshape((V-T,144,176,1))
Cr_x_test = [X[i][V:][:,:,:,2].reshape((10856-V,144,176,1)) for i in range(PAIR_NUM)]
Cr_y_test = Y[V:][:,:,:,2].reshape((10856-V,144,176,1))

print("Training")
modelY = model3(PAIR_NUM)
stop = []
stop.append(EarlyStopping(monitor='val_loss', patience=10, mode='min'))
stop.append(ModelCheckpoint("Y_best.h5", monitor='val_loss', save_best_only=True, mode='min', period=1))
historyY = modelY.fit(Y_x_train, Y_y_train, batch_size=64, nb_epoch=100, callbacks=stop, validation_data=(Y_x_test, Y_y_test))
print("Saving model")
modelY.save('Y_autoencoder_10.h5')

# modelCb = model(PAIR_NUM)
# historyCb = modelCb.fit(Cb_x_train, Cb_y_train, nb_epoch=50, validation_data=(Cb_x_valid, Cb_y_valid))
# print("Saving model")
# modelCb.save('Cb_autoencoder.h5')

# modelCr = model(PAIR_NUM)
# historyCr = modelCr.fit(Cr_x_train, Cr_y_train, nb_epoch=50, validation_data=(Cr_x_valid, Cr_y_valid))
# print("Saving model")
# modelCr.save('Cr_autoencoder.h5')













