import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils import thresh_filter, abs_sobel_thresh, s_filter, sobel_filter, combine_binaries

def show_calibration_result(mtx, dist):

    m = 3

    images = glob.glob('camera_cal/calibration*.jpg')

    for fname in images:
        img = mpimg.imread(fname)

        dst = cv2.undistort(img, mtx, dist, None, mtx)

        fig = plt.figure(figsize=(16, 12))
        plt.subplot(121)
        plt.imshow(img)
        plt.title('Before')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(dst)
        plt.title('After')
        plt.axis('off')
        fig.tight_layout()
        plt.show()
        m -= 1
        if m == 0:
            break

def show_perspective_transform_result(mtx, dist, M):

    m = 3

    images = glob.glob('test_images/*.jpg')

    for fname in images:
        img = mpimg.imread(fname)

        undist = cv2.undistort(img, mtx, dist, None, mtx)

        warped = cv2.warpPerspective(undist, M, (undist.shape[1], undist.shape[0]), flags=cv2.INTER_LINEAR)

        fig = plt.figure(figsize=(20, 12))
        plt.subplot(121)
        plt.imshow(undist)
        plt.title('Before('+fname+')')
        plt.subplot(122)
        plt.imshow(warped)
        plt.title('Perspective transform')
        plt.axis('off')
        fig.tight_layout()
        plt.show()

        m -= 1
        if m == 0:
            break

def show_color_threshold_result(mtx, dist, M):

    m = 3

    images = glob.glob('test_images/*.jpg')

    for fname in images:
        img = mpimg.imread(fname)

        undist = cv2.undistort(img, mtx, dist, None, mtx)

        s_binary = s_filter(undist)
        sxbinary = sobel_filter(undist)
        r_binary = thresh_filter(undist[:, :, 0], (110, 255))

        color_binary = np.dstack(( np.zeros_like(s_binary), sxbinary*200, s_binary*200))

        combined_binary = combine_binaries(s_binary, sxbinary, r_binary)

        fig = plt.figure(figsize=(20, 12))

        plt.subplot(121)
        plt.imshow(color_binary)
        plt.title('color ('+fname+')')

        plt.subplot(122)
        plt.imshow(combined_binary, cmap='gray')
        plt.title('combined')

        plt.axis('off')
        fig.tight_layout()
        plt.show()

        m -= 1
        if m == 0:
            break

def show_filtered_warped_image(mtx, dist, M):

    m = 3

    images = glob.glob('test_images/*.jpg')

    for fname in images:
        img = mpimg.imread(fname)

        undist = cv2.undistort(img, mtx, dist, None, mtx)

        s_binary = s_filter(undist)
        sxbinary = sobel_filter(undist)
        r_binary = thresh_filter(undist[:, :, 0], (110, 255))

        combined_binary = combine_binaries(s_binary, sxbinary, r_binary)

        warped = cv2.warpPerspective(combined_binary, M, (combined_binary.shape[1], combined_binary.shape[0]), flags=cv2.INTER_LINEAR)

        fig = plt.figure(figsize=(20, 12))

        plt.subplot(121)
        plt.imshow(combined_binary, cmap='gray')
        plt.title('combined ('+fname+')')

        plt.subplot(122)
        plt.imshow(warped, cmap='gray')
        plt.title('warped')

        plt.axis('off')
        fig.tight_layout()
        plt.show()

        m -= 1
        if m == 0:
            break
