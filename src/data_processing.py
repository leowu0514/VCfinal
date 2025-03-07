import numpy as np
import glob
import os
from functools import reduce

video_shape = (144, 176)
frame_size = video_shape[0] * video_shape[1]
frame_data_size = int(frame_size * 1.5)
subsample_shape = (video_shape[0]//2, video_shape[1]//2)

DATA_DIR = "../data/"


def load_yuv(path):
    with open(path, "rb") as fin:
        data = fin.read()

    frame_num = len(data) // frame_data_size
    
    data = bytearray(data)
    data = np.array(data, dtype=np.uint8)
    data = data.reshape([frame_num, -1])
    
    Y = data[:, :frame_size].reshape([frame_num, video_shape[0], video_shape[1]])
    U = data[:, frame_size:frame_size*5//4].reshape([frame_num, subsample_shape[0], subsample_shape[1]]).repeat(2, axis=2).repeat(2, axis=1)
    V = data[:, frame_size*5//4:frame_size*3//2].reshape([frame_num, subsample_shape[0], subsample_shape[1]]).repeat(2, axis=2).repeat(2, axis=1)

    yuv_array = np.zeros([frame_num, *video_shape, 3])
    yuv_array[:, :, :, 0] = Y
    yuv_array[:, :, :, 1] = U
    yuv_array[:, :, :, 2] = V

    return yuv_array


def load_all_yuv(data_dir):
    yuv_path_list = sorted(glob.glob(os.path.join(data_dir, "*.yuv")))
    yuv_data_list = []
    for yuv_path in yuv_path_list:
        yuv_data_list.append(load_yuv(yuv_path))

    return yuv_data_list



def yuv_to_rgb(yuv):
    rgb = np.zeros_like(yuv)
    rgb[..., 0] = yuv[..., 0] + 1.402 * (yuv[..., 2] - 128)
    rgb[..., 1] = yuv[..., 0] - 0.344 * (yuv[..., 1] - 128) - 0.714 * (yuv[..., 2] - 128)
    rgb[..., 2] = yuv[..., 0] + 1.772 * (yuv[..., 1] - 128)
                                                 
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def yuv_arr_to_yuv_file(yuv, filename):
    frame_num = yuv.shape[0]
    yuv = yuv.round().astype(np.uint8)

    Y = yuv[..., 0]
    U = yuv[:, ::2, ::2, 1]
    V = yuv[:, ::2, ::2, 2]

    Y = Y.reshape([frame_num, -1])
    U = U.reshape([frame_num, -1])
    V = V.reshape([frame_num, -1])

    yuv = np.concatenate([Y, U, V], axis=1).reshape([-1])

    with open(filename, "wb") as fout:
        fout.write(bytes(list(yuv)))

    return

def compute_PSNR(in1, in2):
    in1 = yuv_to_rgb(in1)
    in2 = yuv_to_rgb(in2)

    diff = in1 - in2
    mse = np.square(diff).sum() / reduce(lambda x, y: x*y, in1.shape, 1)
    PSNR = 10 * np.log10(255 * 255 / mse)
    return PSNR
