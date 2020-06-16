import cv2
from camera_calibration_utility import CameraCalibrationUtility
from tools import Tools
from image_processing_utilities import ImageProcessingUtilities
from video_stream_processing import VideoStreamProcessing
from line import Line


print("Start processing images.")

cameraCalibration = CameraCalibrationUtility()
tools = Tools()
imageProcessing = ImageProcessingUtilities()
videoStreamProcessing = VideoStreamProcessing()
left_right_lines = [Line(), Line()]
global mtx, dist, transform_mtx, inverse_transform_mtx, width, height


def process_frame(frame, im_name="No name"):
    """
    :param frame: a colored image to be processed. width, height should correspond those from calibration.
    :param im_name: optional parameter for plotting with matplotlib.pyplot
    :return: an output image (np.array)
    This function is the image processing pipeline.
    Pre-requisites are the outputs from the camera calibration and image 3D-2D transformation.
    """

    global mtx, dist, transform_mtx, inverse_transform_mtx, width, height, left_right_lines
    # Step 1: distortion correction of the image
    undistorted = cv2.undistort(frame, mtx, dist, None, mtx)

    # Step 2: Color/gradient threshold
    combined_binary_image = imageProcessing.apply_color_gradient_thresholds(undistorted, im_name)

    # Step 3. Perspective transform
    warped_image = cv2.warpPerspective(combined_binary_image, transform_mtx, (width, height))
    # tools.plot_images(test_image, combined_binary_image, warped_image, [im_name,
    #                                                               'combined_binary_image', 'warped_image'])

    # optional histogram visualization step.
    # histogram = tools.get_histogram(warped_image)

    # Step 4. Detect lane lines
    # find a lane line using sliding window only on the first frame.
    # Reuse left and right Polynomials from the first frame for all the consequent frames
    # Find lanes using sliding window method only if:
    # - this is a first frame, hence no lanes have been detected
    # - at least one of the lanes haven't been detected in the previous frame.
    if left_right_lines[0].detected is False and left_right_lines[1].detected is False:
        # Apply Sliding Window and Fit a Polynomial
        # No need to pass left_right_lines, because the line haven't been detected in the previous frame anyway
        out_img, left_right_lines = imageProcessing.fit_sliding_polynomial(warped_image, im_name)
    else:
        # Processing of the 2nd and all consequent frames in the video stream
        out_img, left_right_lines = videoStreamProcessing.search_around_poly(warped_image, left_right_lines)

    # Step 5. Determine the lane curvature in pixels for both lane lines
    ploty, left_right_lines[0].radius_of_curvature, left_right_lines[1].radius_of_curvature = \
        imageProcessing.measure_curvature_pixels(left_right_lines[0].best_fit, left_right_lines[1].best_fit)

    # Auxiliary step: add curvature and offset information to the output image
    text_lines = tools.get_text_overlay(left_right_lines[0].radius_of_curvature,
                                        left_right_lines[1].radius_of_curvature,
                                        left_right_lines[0].line_base_pos)
    output_image = imageProcessing.draw_plane_over_image(ploty, frame, undistorted, warped_image,
                                                         left_right_lines[0].bestx, left_right_lines[1].bestx,
                                                         inverse_transform_mtx, text_lines, im_name)
    return output_image


# Execution flow of the advanced lane finding pipeline
# Definition of the variables
def main():
    global mtx, dist, transform_mtx, inverse_transform_mtx, width, height

    # Step 1: camera calibration and generation of transformation matrices
    project_path = "./"
    calibration_images_path = "camera_cal/"
    name_pattern = 'calibration*.jpg'

    path = project_path+calibration_images_path
    images, im_names = tools.get_image_from_dir(path, name_pattern)
    mtx, dist = cameraCalibration.calibrate_camera(images, im_names)

    test_images_path = "test_images/"
    path = project_path+test_images_path
    name_pattern = '*.jpg'
    images, im_names = tools.get_image_from_dir(path, name_pattern)

    transform_mtx, inverse_transform_mtx, width, height = imageProcessing.get_transformation_values(images, im_names)

    # Optional step: saving calibration results for later re-use
    tools.dump_parameters(mtx, dist, transform_mtx, inverse_transform_mtx, width, height)

    # Step 2: single images testing pipeline
    for pos, test_image in enumerate(images):
        im_name = im_names[pos].split('/')[-1].split('.')[0]
        process_frame(test_image, im_name)

    # Step 3: video stream processing pipeline
    videoStreamProcessing.process_video(process_frame)


# Program start
if __name__ == "__main__":
    main()
