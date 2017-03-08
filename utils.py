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
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

def thresh_filter(s, thresh=(0, 255)):
    binary_output = np.zeros_like(s)
    binary_output[(s > thresh[0]) & (s <= thresh[1])] = 1
    return binary_output

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(channel, orient='x', abs_sobel_thresh_min=0, abs_sobel_thresh_max=255):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 0, 1))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= abs_sobel_thresh_min) & (scaled_sobel <= abs_sobel_thresh_max)] = 1

    # Return the result
    return binary_output

def s_filter(rgb_img, thresh_low=90, thresh_high=255):
    '''
    Input: RGB image
    Output: Binary image
    '''
    hls = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]
    s_binary = thresh_filter(s, (thresh_low, thresh_high))

    return s_binary

def sobel_filter(rgb_img, thresh_low=20, thresh_high=100):
    '''
    Input: RGB image
    Output: Binary image
    '''
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    sxbinary = abs_sobel_thresh(gray, 'x', thresh_low, thresh_high)

    return sxbinary

def combine_binaries(s_img, sx_img, r_img):
    assert s_img.shape == sx_img.shape
    assert sx_img.shape == r_img.shape
    combined_binary = np.zeros_like(s_img)
    combined_binary[((s_img == 1) & (sx_img == 1)) |
                    ((r_img == 1) & (sx_img == 1)) |
                    ((s_img == 1) & (r_img == 1))] = 1
    return combined_binary

def draw_result(undist, warped, lane_line, left_fit, right_fit, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    # cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # color_warp_laneline = cv2.addWeighted(color_warp, 0.8, lane_line, 0.2, 0)
    color_warp_laneline = lane_line
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp_laneline, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result
    # plt.imshow(result)
