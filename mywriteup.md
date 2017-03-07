##Advanced Lane Finding Project
###It's a very interesting project that I never thought I can finish it with my limited computer vision skill. It clearly defined 8 steps to follow so I will explained how I did it with that order.
---

### Camera Calibration

OpenCV has already provided handy functions for camera calibrating. Udacity also provided dozens quality chessboard images. The idea was to collect the 3D points in real world space and 2D points in image plane. Then use cv2.calibrateCamera() to generate the calibration matrix. I saved the mtx and dist to distort.p for later use.

```python
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
```

![alt text][image1]
![alt text][image2]

### Perspective transform

Next, we need to transform the front camera image to bird view image. It will be easier to see the curvature of the land lines.

I added 9 marks in the images indicating the coordinates I used to do the transforming. The up-arrow, down-arrow, cross and square signs in the inner trapezoid transformed to the outer corresponding signs. Those numbers were manually tuned, no serious theory behind. I wrapped it into get_perspective_transform() as below. The return matrix M will be used with the binary image I made in later stages.

![alt text][image3]

```python
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
```

#### Here are the result
![alt text][image4]
![alt text][image5]

### Use different colorspace and apply gradient / Sobel filters to make clear lane lines

The yellow lane lines on concrete road are not as obvious as white lane lines on asphalt road. Needless to say the conditions w/ tree shadows, it's hard to identify the lane lines. So I firstly convert the original RGB colorspace to HSL colorspace. Extract the points from the image with S channel value ranges from 90 to 255. Secondly apply sobel x filters on grayscale image, extract the points with value range from 20 to 100. Then combines these two images into one(using OR).

![alt text][image6]

### Apply perspective transform on the generated binary image

Use the perspective transform matrix M and the combined binary image from last step together to generate a B/W land lines.

![alt text][image7]

### Fit poly

Use fit_poly() in poly.py to find the polynomial coefficients of lane lines. Firstly uses the histogram to find the start point from bottom. And uses sliding window method to identify pixels on the lane lines. Then use numpy's polyfit() to get the coefficients.

After got the polynomial coefficients, calculate the curvature. Uses following parameters to convert pixels to real world distance in meters.

```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

### Coloring the warped image

In the process_image(), I changed the pixels color to red in the left lane line(lineline[0].allx, ally). And use same method to change pixels color to blue in the right lane line. I use fillPoly() to fill green in the road area. After that, Unwarp the colored image to original perspective.

### Add overlay

Add the curvature/offset information to the unwarped image and add it to original frame to generate another video file.

### result

Here is a [link to my video result](./output_images/output.mp4)

Here is a [link to my video result](https://youtu.be/sqAMu2re8mE)

---

###Discussion

#### 1. I spent too much time on testing color/HSL/sobel thresholds. I still didn't get a robust result. My current parameters would fail in the sections with a lot of tree shadow areas. More efforts could take to make a better result.

#### 2. It also took me a while to be familiar with numpy's array manipulation. These are pretty powerful that can make codes more concise.

```python
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))
```

[//]: # (Image References)

[image1]: ./output_images/chessboard-1.png "Chessboard-1"
[image2]: ./output_images/chessboard-2.png "Chessboard-2"
[image3]: ./output_images/perspective_transform-mark.png "Perspective transform-mark"
[image4]: ./output_images/perspective_transform-1.png "Perspective transform-1"
[image5]: ./output_images/perspective_transform-2.png "Perspective transform-2"
[image6]: ./output_images/colorspace-1.png "Colorspace-1"
[image7]: ./output_images/color_warped-1.png "Colorspace warped-1"
