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

The yellow lane lines on concrete road are not as obvious as white lane lines on asphalt road. Needless to say the conditions w/ tree shadows, it's hard to identify the lane lines. So I firstly convert the original RGB colorspace to HSL colorspace. Extract the points from the image with value ranges from 170 to 255. Secondly apply sobel x filters on grayscale image, extract the points with value range from 20 to 100. Then combines these two images into one(using OR).

![alt text][image6]

### Apply perspective transform on the generated binary image

Use the 

[//]: # (Image References)

[image1]: ./output_images/chessboard-1.png "Chessboard-1"
[image2]: ./output_images/chessboard-2.png "Chessboard-2"
[image3]: ./output_images/perspective_transform-mark.png "Perspective transform-mark"
[image4]: ./output_images/perspective_transform-1.png "Perspective transform-1"
[image5]: ./output_images/perspective_transform-1.png "Perspective transform-2"
[image6]: ./output_images/colorspace-1.png "Colorspace-1"


---
End
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.




## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
