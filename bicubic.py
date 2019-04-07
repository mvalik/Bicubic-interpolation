import cv2
import numpy as np
import math
import sys
import time


# Interpolation kernel
def u(s, a):
    abs_s = abs(s)
    if 0 <= abs_s <= 1:
        return (a + 2) * (abs_s ** 3) - (a + 3) * (abs_s ** 2) + 1
    elif 1 < abs_s <= 2:
        return a * (abs_s ** 3) - (5 * a) * (abs_s ** 2) + (8 * a) * abs_s - 4 * a
    return 0


# Paddnig
def padding(img, H, W, C):
    zimg = np.zeros((H + 4, W + 4, C))
    zimg[2:H + 2, 2:W + 2, :C] = img
    # Pad the first/last two col and row
    zimg[2:H + 2, 0:2, :C] = img[:, 0:1, :C]
    zimg[H + 2:H + 4, 2:W + 2, :] = img[H - 1:H, :, :]
    zimg[2:H + 2, W + 2:W + 4, :] = img[:, W - 1:W, :]
    zimg[0:2, 2:W + 2, :C] = img[0:1, :, :C]
    # Pad the missing eight points
    zimg[0:2, 0:2, :C] = img[0, 0, :C]
    zimg[H + 2:H + 4, 0:2, :C] = img[H - 1, 0, :C]
    zimg[H + 2:H + 4, W + 2:W + 4, :C] = img[H - 1, W - 1, :C]
    zimg[0:2, W + 2:W + 4, :C] = img[0, W - 1, :C]
    return zimg


# https://github.com/yunabe/codelab/blob/master/misc/terminal_progressbar/progress.py
def get_progressbar_str(progress, max_len=30):
    bar_len = int(max_len * progress)
    progr_part = '=' * bar_len
    progr_marker = '>' if bar_len < max_len else ''
    progr_empty = ' ' * (max_len - bar_len)
    return f'Progress:[{progr_part}{progr_marker}{progr_empty}] {progress * 100:.1f}%'


# Bicubic operation
def bicubic(img, ratio, a):
    # Get image size
    H, W, C = img.shape

    img = padding(img, H, W, C)
    # Create new image
    dH = math.floor(H * ratio)
    dW = math.floor(W * ratio)
    dst = np.zeros((dH, dW, 3))

    h = 1 / ratio

    print('Start bicubic interpolation')
    print('It will take a little while...')
    inc = 0
    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                x, y = i * h + 2, j * h + 2

                xdec, xint = math.modf(x)
                xds = [abs(xdec - k) for k in range(-1, 3)]
                xint = int(xint)
                ydec, yint = math.modf(y)
                yds = [abs(ydec - k) for k in range(-1, 3)]
                yint = int(yint)

                mat_l = np.matrix([[u(xd, a) for xd in xds]])
                mat_m = np.transpose(np.matrix(img[yint - 1:yint + 3, xint - 1:xint + 3, c]))
                mat_r = np.matrix([[u(yd, a)] for yd in yds])

                dst[j, i, c] = (mat_l @ mat_m) @ mat_r

                # Print progress
                inc = inc + 1
                if inc % 100 == 1:
                    print(f'\r\033[K{get_progressbar_str(inc / (C * dH * dW))}', file = sys.stderr, end='')
    print(f'\r\033[K{get_progressbar_str(1.0)}', file=sys.stderr)
    return dst


if __name__ == '__main__':
    # Read image
    orig_img = cv2.imread('butterfly.png')

    # Scale factor
    ratio = 2
    # Coefficient
    a = -1 / 2

    start_time = time.time()

    result_img = bicubic(orig_img, ratio, a)
    print('Completed!')

    stop_time = time.time()
    print(f'Elapsed time: {stop_time - start_time:.3f}s')

    cv2.imwrite('bicubic_butterfly.png', result_img)
