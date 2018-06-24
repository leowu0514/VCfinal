import os
import sys
import numpy as np
from data_processing import *
from keras.layers import Input, Dense, Dropout, Reshape, dot, Permute, average, concatenate, Flatten, Activation
from keras.layers import BatchNormalization, Conv2D, ZeroPadding2D, AveragePooling2D, MaxPooling2D, UpSampling2D, GRU
from keras.models import Model, Sequential, load_model
from keras.optimizers import RMSprop, Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint

def model_ratio2(pair):
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
	weight = concatenate(flat, axis=1)
	# merge = average(tmp)
	# print(weight.shape, frames.shape)
	merge = dot([weight, frames], 1)
	# print(merge.shape)
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

def model_ratio4(pair):
	inp = [Input((144, 176, 1)) for i in range(pair)]
	tmp = [[] for i in range(pair)]
	flat = [[] for i in range(pair)]
	for i in range(pair):
		# CNNencoder
		x = Conv2D(16, (3,3), activation='relu', padding='same')(inp[i]) # 144 x 176 x 16
		x = MaxPooling2D((2,2), padding='same')(x) # 72 x 88 x 16
		x = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 72 x 88 x 8
		x = MaxPooling2D((2,2), padding='same')(x) # 36 x 44 x 8
		tmp[i] = Conv2D(4, (3,3), activation='relu', padding='same')(x) # 36 x 44 x 4
		tmp[i] = Reshape((1,36*44*4))(tmp[i])
		flat[i] = Flatten()(tmp[i])
		flat[i] = Dense(1)(flat[i])
	
	frames = concatenate(tmp, axis=1)
	weight = concatenate(flat, axis=1)
	# merge = average(tmp)
	# print(weight.shape, frames.shape)
	merge = dot([weight, frames], 1)
	# print(merge.shape)
	merge = Reshape((36,44,4))(merge)

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

def model_ratio8(pair):
	inp = [Input((144, 176, 1)) for i in range(pair)]
	tmp = [[] for i in range(pair)]
	flat = [[] for i in range(pair)]
	for i in range(pair):
		# CNNencoder
		x = Conv2D(32, (3,3), activation='relu', padding='same')(inp[i]) # 144 x 176 x 32
		x = BatchNormalization()(x)
		x = MaxPooling2D((2,2), padding='same')(x)
		
		x = Conv2D(16, (3,3), activation='relu', padding='same')(x) # 72 x 88 x 16
		x = BatchNormalization()(x)
		x = MaxPooling2D((2,2), padding='same')(x)
		
		x = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 36 x 44 x 8
		x = BatchNormalization()(x)
		x = MaxPooling2D((2,2), padding='same')(x)

		tmp[i] = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 18 x 22 x 8
		tmp[i] = Reshape((1,18*22*8))(tmp[i])
		flat[i] = Flatten()(tmp[i])
		flat[i] = Dense(1)(flat[i])
	
	frames = concatenate(tmp, axis=1)
	weight = concatenate(flat, axis=1)
	# merge = average(tmp)
	# print(weight.shape, frames.shape)
	merge = dot([weight, frames], 1)
	# print(merge.shape)
	merge = Reshape((18,22,8))(merge)

	# CNNdecoder
	x = UpSampling2D((2,2))(merge)
	x = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 36 x 44 x 8
	x = UpSampling2D((2,2))(x)
	x = Conv2D(16, (3,3), activation='relu', padding='same')(x) # 72 x 88 x 16
	x = UpSampling2D((2,2))(x)
	out = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x) # 144 x 176 x 1

	# print(inp)

	autoencoder = Model(inp, out)
	autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
	# autoencoder.compile(optimizer=RMSprop(), loss='mean_squared_error')
	return autoencoder

def model_ratio16(pair):
	inp = [Input((144, 176, 1)) for i in range(pair)]
	tmp = [[] for i in range(pair)]
	flat = [[] for i in range(pair)]
	for i in range(pair):
		# CNNencoder
		x = Conv2D(32, (3,3), activation='relu', padding='same')(inp[i]) # 144 x 176 x 32
		x = BatchNormalization()(x)
		x = MaxPooling2D((2,2), padding='same')(x)
		
		x = Conv2D(16, (3,3), activation='relu', padding='same')(x) # 72 x 88 x 16
		x = BatchNormalization()(x)
		x = MaxPooling2D((2,2), padding='same')(x)
		
		x = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 36 x 44 x 8
		x = BatchNormalization()(x)
		x = MaxPooling2D((2,2), padding='same')(x)

		tmp[i] = Conv2D(4, (3,3), activation='relu', padding='same')(x) # 18 x 22 x 4
		tmp[i] = Reshape((1,18*22*4))(tmp[i])
		flat[i] = Flatten()(tmp[i])
		flat[i] = Dense(1)(flat[i])
	
	frames = concatenate(tmp, axis=1)
	weight = concatenate(flat, axis=1)
	# merge = average(tmp)
	# print(weight.shape, frames.shape)
	merge = dot([weight, frames], 1)
	# print(merge.shape)
	merge = Reshape((18,22,4))(merge)

	# CNNdecoder
	x = UpSampling2D((2,2))(merge)
	x = Conv2D(8, (3,3), activation='relu', padding='same')(x) # 36 x 44 x 8
	x = UpSampling2D((2,2))(x)
	x = Conv2D(16, (3,3), activation='relu', padding='same')(x) # 72 x 88 x 16
	x = UpSampling2D((2,2))(x)
	out = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x) # 144 x 176 x 1

	# print(inp)

	autoencoder = Model(inp, out)
	autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
	# autoencoder.compile(optimizer=RMSprop(), loss='mean_squared_error')
	return autoencoder

if len(sys.argv) < 4:
	print("Usage: python train.py <Y_ratio> <Cb_ratio> <Cr_ratio>")
	print("-- <ratio> can be 2, 4, 8 or 16!")
	sys.exit()
try:
	Y_ratio = sys.argv[1]
	Cb_ratio = sys.argv[2]
	Cr_ratio = sys.argv[3]
except:
	print("python train.py <Y_ratio> <Cb_ratio> <Cr_ratio>")
	print("-- <ratio> can be 2, 4, 8 or 16!")
	sys.exit()

PAIR_NUM = 3
print("Loading data")
try:
	data = load_all_yuv("../data/")
except:
	print("no data exist!")
	sys.exit()

position = np.zeros(len(data)).astype(int)
tmp = 0
for i in range(len(data)):
	position[i] = tmp
	tmp += data[i].shape[0]

# normalize
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

print("Training Y")
modelY = []
if Y_ratio == 2:
	modelY = model_ratio2(PAIR_NUM)
elif Y_ratio == 4:
	modelY = model_ratio4(PAIR_NUM)
elif Y_ratio == 8:
	modelY = model_ratio8(PAIR_NUM)
elif Y_ratio == 16:
	modelY = model_ratio16(PAIR_NUM)
else:
	print("Unsupported ratio! using ratio 16 instead.")
	modelY = model_ratio16(PAIR_NUM)

# modelY = model_ratio4(PAIR_NUM)
stop = []
stop.append(EarlyStopping(monitor='val_loss', patience=10, mode='min'))
stop.append(ModelCheckpoint("Y_best.h5", monitor='val_loss', save_best_only=True, mode='min', period=1))
historyY = modelY.fit(Y_x_train, Y_y_train, batch_size=64, nb_epoch=100, callbacks=stop, validation_data=(Y_x_test, Y_y_test))
print("Saving Y model")
modelY.save('Y.h5')

print("Training Cb")
modelCb = []
if Cb_ratio == 2:
	modelCb = model_ratio2(PAIR_NUM)
elif Cb_ratio == 4:
	modelCb = model_ratio4(PAIR_NUM)
elif Cb_ratio == 8:
	modelCb = model_ratio8(PAIR_NUM)
elif Cb_ratio == 16:
	modelCb = model_ratio16(PAIR_NUM)
else:
	print("Unsupported ratio! using ratio 16 instead.")
	modelCb = model_ratio16(PAIR_NUM)

# modelCb = model_ratio8(PAIR_NUM)
stop = []
stop.append(EarlyStopping(monitor='val_loss', patience=10, mode='min'))
stop.append(ModelCheckpoint("Cb_best.h5", monitor='val_loss', save_best_only=True, mode='min', period=1))
historyCb = modelCb.fit(Cb_x_train, Cb_y_train, batch_size=64, nb_epoch=100, validation_data=(Cb_x_test, Cb_y_test))
print("Saving Cb model")
modelCb.save('Cb.h5')

print("Training Cr")
modelCr = []
if Cr_ratio == 2:
	modelCr = model_ratio2(PAIR_NUM)
elif Cr_ratio == 4:
	modelCr = model_ratio4(PAIR_NUM)
elif Cr_ratio == 8:
	modelCr = model_ratio8(PAIR_NUM)
elif Cr_ratio == 16:
	modelCr = model_ratio16(PAIR_NUM)
else:
	print("Unsupported ratio! using ratio 16 instead.")
	modelCr = model_ratio16(PAIR_NUM)

# modelCr = model_ratio8(PAIR_NUM)
stop = []
stop.append(EarlyStopping(monitor='val_loss', patience=10, mode='min'))
stop.append(ModelCheckpoint("Cr_best.h5", monitor='val_loss', save_best_only=True, mode='min', period=1))
historyCr = modelCr.fit(Cr_x_train, Cr_y_train, batch_size=64, nb_epoch=100, validation_data=(Cr_x_test, Cr_y_test))
print("Saving Cr model")
modelCr.save('Cr.h5')













