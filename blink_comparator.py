import os
from pathlib import Path
import numpy as np
import cv2 as cv

MIN_NUM_KEYPOINT_MATCHES = 50  # constant for minimum number of keypoint matches


def main():
    """loop through 2 folders with paired images, register & blink images."""
    night1_files = sorted(os.listdir(
        'C:/Users/austi/Documents/pythonWork/findingz_pluto_with_open_cv_imag_ analysis/night_1'))
    # create sorted list of filenames for night 1 folder
    night2_files = sorted(os.listdir(
        'C:/Users/austi/Documents/pythonWork/findingz_pluto_with_open_cv_imag_ analysis/night_2'))
    # create sorted list of filenames for night 2 folder
    path1 = Path.cwd() / 'C:/Users/austi/Documents/pythonWork/findingz_pluto_with_open_cv_imag_ analysis/night_1'
    # assign path class name for input folder 1
    path2 = Path.cwd() / 'C:/Users/austi/Documents/pythonWork/findingz_pluto_with_open_cv_imag_ analysis/night_2'
    # assign path class name for input folder 2
    path3 = Path.cwd() / \
            'C:/Users/austi/Documents/pythonWork/findingz_pluto_with_open_cv_imag_ analysis/night_1_registered'
    # assign class path name for output folder

    for i, _ in enumerate(night1_files):  # create index for files in folder
        img1 = cv.imread(str(path1 / night1_files[i]), cv.IMREAD_GRAYSCALE)  # read image 1 in greyscale
        img2 = cv.imread(str(path2 / night2_files[i]), cv.IMREAD_GRAYSCALE)  # read image 2 in greyscale
        print("Comparing {} to {}.\n".format(night1_files[i], night2_files[i]))  # display status for comparing
        kp1, kp2, best_matches = find_best_matches(img1, img2)  # find keypoints and best matches with function
        img_match = cv.drawMatches(img1, kp1, img2, kp2,
                                   best_matches, outImg=None)  # draw lines to match image keypoints
        height, width = img1.shape  # get size of image 1
        cv.line(img_match, (width, 0), (width, height), (255, 255, 255), 1)  # draw a line on the right side of image 1
        QC_best_matches(img_match)  # comment out to ignore. displays best matches quality control
        img1_registered = register_image(img1, img2, kp1, kp2, best_matches)  # register first image to second

        blink(img1, img1_registered, 'Check Registration', num_loops=5)  # blink comparator function
        out_filename = '{}_registered.png'.format(night1_files[i][:-4])  # create file path for out image
        cv.imwrite(str(path3 / out_filename), img1_registered)  # Will overwrite! write image to file
        cv.destroyAllWindows()  # destroy all windows to remove clutter
        blink(img1_registered, img2, 'Blink Comparator', num_loops=15)  # call blink again to display only the comparing


def find_best_matches(img1, img2):
    """Return list of keypoints and list of best matches for the two images"""
    orb = cv.ORB_create(nfeatures=100)  # detect keypoint orb objects class instance
    kp1, desc1 = orb.detectAndCompute(img1, mask=None)  # find image 1 keypoints
    kp2, desc2 = orb.detectAndCompute(img2, mask=None)  # find image 2 keypoints
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  # bruteforce matching keypoints in both images instance
    matches = bf.match(desc1, desc2)  # match keypoints in both images
    matches = sorted(matches, key=lambda x: x.distance)  # sort list of keypoint matches by distance
    best_matches = matches[:MIN_NUM_KEYPOINT_MATCHES]  # take 50 best matches

    return kp1, kp2, best_matches


def QC_best_matches(img_match):
    """draw best keypoint matches connected by colored lines"""
    cv.imshow('Best {} Matches'.format(MIN_NUM_KEYPOINT_MATCHES), img_match)  # display the window with keypoint circles
    cv.waitKey(2500)  # keeps window active 2.5 seconds


def register_image(img1, img2, kp1, kp2, best_matches):
    """return first image registered to second image"""
    if len(
            best_matches) >= MIN_NUM_KEYPOINT_MATCHES:
        # if the best matches list is greater than MIN_NUM_KEYPOINT_MATCHES
        src_pts = np.zeros((len(best_matches), 2), dtype=np.float32)  # create array of zeros
        dst_pts = np.zeros((len(best_matches), 2), dtype=np.float32)  # create a row of zeros
        for i, match in enumerate(best_matches):  # data from best matches
            src_pts[i, :] = kp1[match.queryIdx].pt  # fill array with points
            dst_pts[i, :] = kp2[match.queryIdx].pt  # fill array 2 with points
        h_array, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)  # use homography to align images
        height, width = img2.shape  # Get dimensions of image 2
        img1_warped = cv.warpPerspective(img1, h_array, (width, height))  # warp image so it aligns with first image

        return img1_warped

    else:
        print("Warning: Number of keypoint matches < {}\n".format(MIN_NUM_KEYPOINT_MATCHES))
        return img1


def blink(image_1, image_2, window_name, num_loops):
    """Replicate blink comparator with 2 images"""
    for _ in range(num_loops):
        cv.imshow(window_name, image_1)
        cv.waitKey(330)
        cv.imshow(window_name, image_2)
        cv.waitKey(330)


if __name__ == '__main__':
    main()
