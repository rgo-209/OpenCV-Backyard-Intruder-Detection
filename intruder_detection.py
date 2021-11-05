'''
Intruder Detection
=========
Usage
-----
intruder_detection.py [path to image files]

'''

import numpy as np
import cv2
import os
import sys

# define resize factor
RESIZE_FACTOR = 0.3
# define morphological kernel
kernel = np.ones((7, 7), np.uint8)


def intruder_detection(images_path):
    """
        This function is used to detect intruder in the frames
        provided in the given path.
    :param images_path: path to images
    :return:            none
    """

    # create a Background Subtractor MOG2 model
    background_model = cv2.createBackgroundSubtractorMOG2(history=20, detectShadows=False)

    # create a numpy array of colors for segmentation
    colors = np.int32(list(np.ndindex(2, 2, 2))) * 255

    # iterate through all files
    for image_entry in os.scandir(images_path):
        # if the file is an image file
        if image_entry.is_file() and (image_entry.path.endswith(".jpg") or image_entry.path.endswith(".JPG")):
            # read the image
            img = cv2.imread(image_entry.path)
            # retrieve the height, width and number of channels
            height, width, channel = img.shape

            # resize the image to fit the screen
            img = cv2.resize(img, (int(width * RESIZE_FACTOR), int(height * RESIZE_FACTOR)))

            # convert the image to LAB format
            lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            # blur the L channel
            lab_img[:, :, 0] = cv2.blur(lab_img[:, :, 0], (21, 21), 0)
            # blur the L channel again
            lab_img[:, :, 0] = cv2.blur(lab_img[:, :, 0], (21, 21), 0)

            # convert image back to an RGB image
            rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

            # blur the image
            blur_rgb_img = cv2.medianBlur(rgb_img, 7)

            # find the foreground mask
            foreground_mask = background_model.apply(blur_rgb_img)

            # apply dilation to get background mask
            background_mask = cv2.dilate(foreground_mask, kernel, iterations=3)

            # apply distance transform on the foreground mask
            dist_transform = cv2.distanceTransform(foreground_mask, cv2.DIST_L2, 0)

            # apply thresholding on the result of distance transform to get fix foreground
            ret, fix_foreground = cv2.threshold(dist_transform, 0.65 * dist_transform.max(), 255, 0)

            # apply erosion on fix foreground found to suppress noise and find known region
            fix_foreground = cv2.erode(fix_foreground, kernel)
            fix_foreground = np.uint8(fix_foreground)

            # Find unknown region by subtracting the fix background and erosion result
            unknown = cv2.subtract(background_mask, fix_foreground)

            # Generate Marker labelling
            ret2, markers = cv2.connectedComponents(fix_foreground)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1

            # Now, mark the region of unknown with zero
            markers[unknown == 255] = 0

            # call watershed function on image and the markers found
            cv2.watershed(img, markers)

            # create and add segmentation overlay over the image
            overlay = colors[np.minimum(markers, 2)]
            watershed_result = cv2.addWeighted(img, 0.5, overlay, 0.5, 0.0, dtype=cv2.CV_8UC3)

            # display the original image read
            cv2.imshow('Original Image', img)

            # display the mask after detecting movement by uncommenting next line
            # cv2.imshow('Fix Foreground', fix_foreground)

            # display the watershed results
            cv2.imshow('watershed', watershed_result)


            # get waitkey
            k = cv2.waitKey(1) & 0xff

            # exit code if esc is pressed
            if k == 27:
                break

            # save the current frame and watershed result if s is pressed
            elif k == ord('s') or k == ord('S'):
                cv2.imwrite("Output/WatershedOp/op_"+image_entry.name, watershed_result)
                cv2.imwrite("Output/OriginalImg/origin_"+image_entry.name, img)
                print("Saved output to Output/WatershedOp/op_"+image_entry.name)

    cv2.destroyAllWindows()


if __name__ == '__main__' :
    images_path = None
    try :
        # get path to images
        images_path = sys.argv[1]
        print(__doc__)
        print("Running intruder detection on folder: ", images_path, '\n')
        intruder_detection(images_path)
        print("\nFinished.")

    except :
        print('Usage\n-----\nintruder_detection.py [path to image files]')


