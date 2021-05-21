import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import pandas as pd
import json
from scipy.optimize import least_squares
import math
import random
from misc.utils import *
from misc.FundamentalMatrix import *
from misc.PoseEstimation import recoverPose
from misc.printFunctions import *
from misc.DisparityMap import *
import argparse

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--dataset', default='1', help='Dataset type , Default: 1')
    
    Args = Parser.parse_args()
    dataset = int(Args.dataset)

    
    if dataset == 1:
        SavePath = './Output/Dataset1/'
        d_thresh = 100000
    if dataset == 2:
        SavePath = './Output/Dataset2/'
        d_thresh = 90000
    if dataset == 3:
        SavePath = './Output/Dataset3/'
        d_thresh = 200000
        
    images, K1, K2, params = readData(dataset, BasePath = "../Data/Project 3/")    
    foldercheck(SavePath)
    im1, im2 = images
    im1, im2 = rgb(rescale(im1,30)), rgb(rescale(im2,30))
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    pts1, pts2, im_matches = SIFTpoints(im1, im2)
    print('SIFT points Detected : ', len(pts1))
    
    data = (pts1,pts2)
    F,inlier_mask = FundamentalMatrix(data ,s = 8, thresh = 0.001,n_iterations = 2000)
    pts1_ = pts1[inlier_mask==1]
    pts2_ = pts2[inlier_mask==1]    

    E = EssentialMatrix(K1,K2, F)
    print('Inliers SIFT points : ', len(pts1_))
    
    R, C, x3D = recoverPose(E, pts1_, pts2_, K1, K2)
        
    l1 = cv2.computeCorrespondEpilines(pts2_.reshape(-1,1,2), 2,F)
    im2_epilines ,_ = drawEpilines(im1,im2,l1[:10],pts1_,pts2_)
    l2 = cv2.computeCorrespondEpilines(pts1_.reshape(-1,1,2), 1,F)
    im1_epilines,_ = drawEpilines(im2,im1,l2[:10],pts2_,pts1_)
    out = np.hstack((im1_epilines, im2_epilines))
    cv2.imwrite(SavePath+'epilinesImage1.png', out)
    
    
    
    ret,H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1_), np.float32(pts2_), F, imgSize=(w1, h1))
    im1_rectified = cv2.warpPerspective(im1, H1, (w1, h1))
    im2_rectified = cv2.warpPerspective(im2, H2, (w2, h2))
    out = np.hstack((im1_rectified, im2_rectified))
    cv2.imwrite(SavePath+'rectifiedImage.png', out)
    
    dst1 = cv2.perspectiveTransform(pts1_.reshape(-1,1,2), H1).squeeze()
    dst2 = cv2.perspectiveTransform(pts2_.reshape(-1,1,2),H2).squeeze()
    lines1_ = epiLines(pts2_,2, F, w2)
    warpedlines1 = warpEpilines(lines1_, H1)
    lines2_ = epiLines(pts1_,1, F, w2)
    warpedlines2 = warpEpilines(lines2_, H2)
    im1_print = drawLines(im1_rectified, warpedlines1[:10], dst1[:10])
    im2_print = drawLines(im2_rectified, warpedlines2[:10], dst2[:10])
    out = np.hstack((im1_print, im2_print))
    cv2.imwrite(SavePath+'epilines_rectifiedImage.png', out)
    
    imL, imR = gray(im1_rectified), gray(im2_rectified)
    disparityMap = DisparityMap(imL, imR, warpedlines1, warpedlines2, win_size = 10, searchRange = 100 )
    np.save(SavePath + 'disparityMap.npy',disparityMap)
#     disparity_map_print = cv2.normalize(disparityMap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    plt.figure(figsize=  (10,10))
    plt.imshow(disparityMap, cmap=plt.cm.RdBu, interpolation='bilinear')
    plt.savefig(SavePath+ ' disparityMap.png')
    plt.imshow(disparityMap, cmap='gray', interpolation='bilinear')
    plt.savefig(SavePath+ ' disparityMap_gray.png')
    
    baseline = params[1]
    f = K1[0,0]
    depthMap = (baseline*f)/(disparityMap + 1e-15)
    depthMap[depthMap > d_thresh] = d_thresh
    depthMap = np.uint8(depthMap * 255 / np.max(depthMap))
    plt.figure(figsize=  (10,10))
    plt.imshow(depthMap, cmap=plt.cm.RdBu, interpolation='bilinear')
    plt.savefig(SavePath+ ' depthMap.png')
    plt.imshow(depthMap, cmap='gray', interpolation='bilinear')
    plt.savefig(SavePath+ ' depthMap_gray.png')
    
    
if __name__ == '__main__':
    main()
    