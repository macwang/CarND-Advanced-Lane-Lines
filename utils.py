import cv2
import glob
import pickle
import numpy as np

def build_distort_p(save_to):
    images = glob.glob('camera_cal/calibration*.jpg')
    if len(images) == 0:
        raise Exception

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save to file
    data = {'mtx': mtx,
            'dist': dist}

    with open(save_to, 'wb') as output:
        pickle.dump(data, output)

def load_distort_p(load_from):
    dist_pickle = pickle.load(open(load_from, "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    return mtx, dist

def get_perspective_transform():
    mid = 1280/2         # mid line
    t1, b1 = 450, 700    # src top, bottom
    t_tw, t_bw = 90, 820 # trapezoid top, bottom
    t2, b2 = 10, 720     # dst top, bottom
    tw = 620             # dst width

    src = np.float32([[mid-t_tw, t1], [mid+t_tw, t1], [mid-t_bw, b1], [mid+t_bw, b1]])
    dst = np.float32([[mid-tw, t2], [mid+tw, t2], [mid-tw, b2], [mid+tw, b2]])

    M = cv2.getPerspectiveTransform(src, dst)

    return M
