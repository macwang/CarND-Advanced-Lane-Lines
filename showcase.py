import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_calibration_result(mtx, dist):

    m = 3

    images = glob.glob('../camera_cal/calibration*.jpg')

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

    images = glob.glob('../test_images/*.jpg')

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
