import numpy as np
import glob
import os

video_shape = (144, 176)
frame_size = video_shape[0] * video_shape[1]
frame_data_size = int(frame_size * 1.5)
subsample_shape = (video_shape[0], video_shape[1]//4)

DATA_DIR = "data/"


def load_yuv(path):
    with open(path, "rb") as fin:
        data = fin.read()
    
    frame_num = len(data) // frame_data_size
    
    data = bytearray(data)
    data = np.array(data, dtype=np.uint8)
    data = data.reshape([frame_num, -1])
    
    Y = data[:, :frame_size].reshape([frame_num, video_shape[0], video_shape[1]])
    Cb = data[:, frame_size:frame_size*5//4].reshape([frame_num, subsample_shape[0], subsample_shape[1]]).repeat(4, axis=2)
    Cr = data[:, frame_size*5//4:frame_size*3//2].reshape([frame_num, subsample_shape[0], subsample_shape[1]]).repeat(4, axis=2)
    
    yuv_array = np.zeros([frame_num, *video_shape, 3])
    yuv_array[:, :, :, 0] = Y
    yuv_array[:, :, :, 1] = Cb
    yuv_array[:, :, :, 2] = Cr
    
    return yuv_array


def load_all_yuv():
    yuv_path_list = sorted(glob.glob(os.path.join(DATA_DIR, "*.yuv")))
    yuv_data_list = []
    for yuv_path in yuv_path_list:
        yuv_data_list.append(load_yuv(yuv_path))

    return yuv_data_list


