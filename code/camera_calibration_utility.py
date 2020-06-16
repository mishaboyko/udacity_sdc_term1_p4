import numpy as np
import cv2
from tools import Tools


class CameraCalibrationUtility:

    def __init__(self):
        # define global variables
        self.NX = 9
        self.NY = 6
        self.tools = Tools()

    def unwarp_corners(self, undist_img, corners):
        src = np.float32([corners[0], corners[self.NX - 1], corners[(self.NX * self.NY) - 1],
                          corners[(self.NX * self.NY) - self.NX]])
        height, width = undist_img.shape  # no color channels
        offset = 100  # offset for dst points
        dst = np.float32(
            [[offset, offset], [width - offset, offset], [width - offset, height - offset], [offset, height - offset]])
        transform_mtx = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undist_img, transform_mtx, (width, height))
        return warped, transform_mtx

    def calibrate_camera(self, images, im_names):

        good_calibration_images = []
        good_calibration_images_grayscale = []

        # 3D points in real space
        objpoints = []
        # 2D points of the corners of all images on the image plane
        good_corners = []
        # prepare object points
        objpts = np.zeros((self.NY*self.NX, 3), np.float32)

        # x and y coordinates
        objpts[:, :2] = np.mgrid[0:self.NX, 0:self.NY].T.reshape(-1, 2)

        for image in images:
            # Convert to grayscale
            grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(grayscaled_image, (self.NX, self.NY), None)

            # If found, add object & image points. Then draw corners
            if ret:
                good_calibration_images.append(image)
                good_calibration_images_grayscale.append(grayscaled_image)
                good_corners.append(corners)
                objpoints.append(objpts)

                # Draw and display the corners
                cv2.drawChessboardCorners(image, (self.NX, self.NY), corners, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, good_corners,
                                                           grayscaled_image.shape[::-1], None, None)

        for pos, image in enumerate(good_calibration_images):
            undist = cv2.undistort(good_calibration_images_grayscale[pos], mtx, dist, None, mtx)
            warped_image, transform_mtx = self.unwarp_corners(undist, good_corners[pos])

            # Plot 3 steps of image calibration
            # self.tools.plot_images(image, undist, warped_image,
            #                        [im_names[pos].split('/')[-1], 'Undistorted Image', 'Warped Image'])

        return mtx, dist
