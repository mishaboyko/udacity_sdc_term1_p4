# Writeup for Advanced Lane Finding Project

# Goals & Steps

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transform and gradient to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and a 2nd-order polynomial fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I consider the rubric points individually and describe how I addressed each point in my implementation.

## Camera Calibration

### 1. Computation of the camera matrix and distortion coefficients

The invocation code for this step is contained in the lines #76-90 of the file called `advanced_lane_lines.py`.  

The execution code of this procedure is stored in file `camera_calibration_utility.py`

This step consists of following sub-steps:

1. Load chessboard calibration images
2. I use them in the function `calibrate_camera()`in file `camera_calibration_utility.py`, lines #25-58.  I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x=9, y=6) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
3. For each calibration image, I convert it to grayscale, search chessboard corners using `cv2.findChessboardCorners()`. If corners are found, I draw them on the original image.
4. I use the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  
5. Finally I apply this distortion correction to the test image using the `cv2.undistort()` function.
6. For representation purposes I also warped all images to 2D perspective.
7. After having all matrices and coefficients, I dump them into pickle file for possible utilization.

Here is the result of this step: 

<img src="./output_images/calibration_images/calibr_1.png" alt="calibr_1" style="zoom:50%;" />

<img src="./output_images/calibration_images/calibr_4.png" alt="calibr_4" style="zoom:50%;" />

<img src="./output_images/calibration_images/calibr_5.png" alt="calibr_5" style="zoom:50%;" />

<img src="./output_images/calibration_images/calibr_6.png" alt="calibr_6" style="zoom:50%;" />

<img src="./output_images/calibration_images/calibr_12.png" alt="calibr_12" style="zoom:50%;" />

<img src="./output_images/calibration_images/calibr_16.png" alt="calibr_16" style="zoom:50%;" />

<img src="./output_images/calibration_images/calibr_17.png" alt="calibr_17" style="zoom:50%;" />

<img src="./output_images/calibration_images/calibr_18.png" alt="calibr_18" style="zoom:50%;" />

<img src="./output_images/calibration_images/calibr_19.png" alt="calibr_19" style="zoom:50%;" />

<img src="./output_images/calibration_images/calibr_20.png" alt="calibr_20" style="zoom:50%;" />

## Pipeline (single test images)

The invocation code for this pipeline is in the lines #95-98, file `advanced_lane_lines.py`.  

The code for execution flow of this procedure can be found in function `process_frame()` (lines #19-68, file `image_processing_utilities.py`).

For test images processing pipeline I actually use the same method, as for video-pipeline, but with a slight change, that I apply sliding window to every image and don't reuse polynomial coefficients of the lane lines in the consequent images. I do it by commenting out condition

```python
if left_right_lines[0].detected is False and left_right_lines[1].detected is False:
    ...
else
	...
```

### 1. Distortion-corrected image

I didn't plot any distortion-corrected images because:

- one doesn't see any difference anyhow
- this step is included in all other consequent images, so distortion-correction can be seen there

### 2. Color transforms & gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

This part is invoked in `apply_color_gradient_thresholds()` (lines #33, file `advanced_lane_lines.py`).

Implementation is in lines #51-106 of the file `image_processing_utilities.py`.

I used a combination of color and gradient thresholds to generate a binary image. Color thresholding steps of the saturation channel of the HLS-color-spaced image are in `threshold_color()` (lines #70 through #83 in `image_processing_utilities.py`). Gradient threshold is applied in `threshold_gradient()` (lines #51 through #67 in `image_processing_utilities.py`).  At the end I combine them together. 

Here's an example of my output for this step with stacked thresholds and applied distortion correction:

<img src="./output_images/color_gradient_thresholded_images/thresh2.png" alt="thresh2" style="zoom:50%;" />

<img src="./output_images/color_gradient_thresholded_images/thresh5.png" alt="thresh5" style="zoom:50%;" />

<img src="./output_images/color_gradient_thresholded_images/thresh_sl2.png" alt="thresh_sl2" style="zoom:50%;" />

### 3. Perspective transform

The code for my perspective transform is invoked in line #36, file `advanced_lane_lines.py`). This function get `transform_mtx` which I have previously obtained with some other parameters during camera calibration phase when invoking function `get_transformation_values()` in line #90, file `advanced_lane_lines.py`.  The implementation of this function can be seen in lines #38-48, file `image_processing_utilities.py`. The actual transformation as well as inverse transformation matrices come from function `get_trapezoidal_transform_matrix()` (lines #14-36, file `image_processing_utilities.py`). Here I also define source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
        src = np.float32(
            [[586, 461],  # top left
             [705, 461],  # top right
             [1041, 676],  # bottom right
             [276, 676]])  # bottom left

        dst = np.float32(
            [[300, 200],  # top left
             [900, 200],  # top right
             [900, 710],  # bottom right
             [300, 710]])  # bottom left
```

To obtain the matrices and verify had-coded points, I have used only 2 images with 2 straight lines.

I verified that my perspective transform works as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image:

<img src="./output_images/trapezoidal_transformed_images/warped_sl1.png" alt="warped_sl1" style="zoom:50%;" />

<img src="./output_images/trapezoidal_transformed_images/warped_sl2.png" alt="warped_sl2" style="zoom:50%;" />

Final perspective transform of the thresholded binary images finally looks like this:

<img src="./output_images/color_gradient_thresholded/straight_lines1.jpgthresh.jpg" alt="straight_lines1.jpgthresh" style="zoom:50%;" />

![straight_lines2.jpgthresh](./output_images/color_gradient_thresholded/straight_lines2.jpgthresh.jpg)

![test1.jpgthresh](./output_images/color_gradient_thresholded/test1.jpgthresh.jpg)

![test2.jpgthresh](./output_images/color_gradient_thresholded/test2.jpgthresh.jpg)

![test3.jpgthresh](./output_images/color_gradient_thresholded/test3.jpgthresh.jpg)

![test4.jpgthresh](./output_images/color_gradient_thresholded/test4.jpgthresh.jpg)

![test5.jpgthresh](./output_images/color_gradient_thresholded/test5.jpgthresh.jpg)

![test6.jpgthresh](./output_images/color_gradient_thresholded/test6.jpgthresh.jpg)

### 4. Identify lane-line pixels and fit their positions with a polynomial

The code for this step is in function `find_lane_pixels()`(lines #109-188, file `image_processing_utilities.py`). I have re-used the exemplary function from the project example. This step is performed only when no lane lines are found, when sliding window for lines search is applied. The code for sliding window can be found in function `fit_sliding_polynomial()`(lines #201-252, file `image_processing_utilities.py`).

Here's the result of this step:

![sl_1](./output_images/lane_lines/sl_1.png)

![sl_2](./output_images/lane_lines/sl_2.png)

![test1](./output_images/lane_lines/test1.png)

![test2](./output_images/lane_lines/test2.png)

![test3](./output_images/lane_lines/test3.png)

![test4](./output_images/lane_lines/test4.png)

![test5](./output_images/lane_lines/test5.png)

![test6](./output_images/lane_lines/test6.png)

### 5. Histograms of the lane line detection

`find_lane_pixels()` function uses histogram to detect starting pixels on the bottom of the image where the lane border lines start. I have also visualized this step for validation (line #41 in `advanced_lane_lines.py`) , but disabled in the pipeline. The implementation is in `function get_histogram()` (lines #83-97, file `tools.py`). Here are the plots for each test image:

![sl_1](./output_images/histograms/sl_1.png)

![sl_2](./output_images/histograms/sl_2.png)

![test1](./output_images/histograms/test1.png)

![test2](./output_images/histograms/test2.png)

![test3](./output_images/histograms/test3.png)

![test4](./output_images/histograms/test4.png)

![test5](./output_images/histograms/test5.png)

![test6](./output_images/histograms/test6.png)

### 6. Radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated radius of curvature for each lane individually and didn't combine them for the sake of transparency.

**Radius of curvature**: `measure_curvature_pixels()` (lines #255-274, file ``image_processing_utilities.py`). The actual code for it:

```
# Calculation of R_curve (radius of curvature)
left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
```

Formula and explanation are taken from https://www.intmath.com/applications-differentiation/8-radius-curvature.php

**Relative vehicle position**: `calculate_distance_from_lane_center()` (lines #190-199, file ``image_processing_utilities.py`). The algorithm is as follows:

1. Assume width of the image in pixel = 1280, lane width in real world = 3.7m and frame middle = 640 pixel.
2. Get the botton pixels for left and right lane lines as a reference for calculation.
3. Calculate actual lane width in pixel from the frame: `lane_width_pixel = right_pixel - left_pixel`
4. Calculate offset from the middle pixel in pixels: `offset_from_middle_pixel = lane_middle_pixel - frame_middle`
5. Convert pixel offset to meters: `offset_from_middle_m = (offset_from_middle_pixel * lane_width_m)/lane_width_pixel`

### 7. Images of my result plotted back down onto the road

This step is invoked in lines #61 through #67 in file `advanced_lane_lines.py`.

The implementation is done in:

* for textual representation of curvature and lane offset: `get_text_overlay()` (lines #100-110, file `tools.py`)
* drawing of text and plane onto original image: `draw_plane_over_image()` (lines #295-318, file `image_processing_utilities.py`)

Here are examples of my result on a test image:

![straight_lines1_overlay](./output_images/lane_plane_overlays/straight_lines1_overlay.jpg)

![straight_lines2_overlay](./output_images/lane_plane_overlays/straight_lines2_overlay.jpg)

![test1_overlay](./output_images/lane_plane_overlays/test1_overlay.jpg)

![test2_overlay](./output_images/lane_plane_overlays/test2_overlay.jpg)

![test3_overlay](./output_images/lane_plane_overlays/test3_overlay.jpg)

![test4_overlay](./output_images/lane_plane_overlays/test4_overlay.jpg)

![test5_overlay](./output_images/lane_plane_overlays/test5_overlay.jpg)

![test6_overlay](./output_images/lane_plane_overlays/test6_overlay.jpg)

---

## Pipeline (video)

Here's a [link to my video result](./output_video/project_video_out.mp4)

---

# Discussion

## 1. Problems / issues you faced in your implementation of this project. Where will my pipeline likely fail?  What could you do to make it more robust?

I didn't implement a recovery mechanism if lanes disappear. **Solution**: Reset lane lines and apply sliding window in the next frame.

My implementation will fail (with solution ideas):

* if the contrast of the lane lines is not good enough (as tested on challenge_video.mp4). **Solution**: fine-tune thresholds for saturation and Sobel operator. Experiment with other color channels.
* There are more parallel vertical lines on the road. **Solution**: fine-tune thresholds for saturation and Sobel operator. Experiment with other color channels.
* Car is crossing the lanes. **Solution**: vehicle offset into lines detection, keep steering angle, observe objects on the road, keep the steering angle, track all parallel lines direction steering and wait until both parallel lane border lines are at their places again.
* Lane curvature makes the algorithm track neighboring lane far ahead. **Solution**: assure left and right lane lines never flip. If so - reset and apply sliding window. Decrease the detecting distance ahead of the vehicle.