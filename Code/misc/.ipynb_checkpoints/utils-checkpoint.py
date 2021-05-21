import cv2
import numpy as np
import glob

import os
                           
def readData(dataset = 1, BasePath = "../Data/Project 3/"):    
    
    if dataset == 1:
        DataPath = BasePath + 'Dataset 1/'
        K1= np.array([[5299.313, 0, 1263.818],
                      [0, 5299.313, 977.763],
                      [0, 0, 1,]])
        K2= np.array([[5299.313, 0, 1438.004],
                      [0, 5299.313, 977.763],
                      [0, 0, 1,]])
        doffs, baseline,width, height, ndisp, isint, vmin, vmax,dyavg, dymax = 174.186, 177.288, 2988, 2008, 180,0, 54, 147, 0,0
        params = (doffs, baseline,width, height, ndisp, isint, vmin, vmax,dyavg, dymax)
        
    elif dataset == 2:
        DataPath = BasePath + 'Dataset 2/'
        K1= np.array([[4396.869, 0, 1353.072],
                      [0, 4396.869, 989.702],
                      [0, 0, 1,]])
    
        K2= np.array([[4396.869, 0, 1538.86],
                      [0, 4396.869, 989.702],
                      [0, 0, 1,]])
        doffs, baseline,width, height, ndisp, isint, vmin, vmax,dyavg, dymax = 185.788, 144.049, 2880, 1980, 640,0, 17, 619, 0,0
        params = (doffs, baseline,width, height, ndisp, isint, vmin, vmax,dyavg, dymax)

    else:
        DataPath = BasePath + 'Dataset 3/'
        K1= np.array([[5806.559, 0, 1429.219],
                      [0, 5806.559, 993.403],
                      [0, 0, 1,]])
    
        K2= np.array([[5806.559, 0, 1543.51],
                      [0, 5806.559, 993.403],
                      [0, 0, 1,]])
        doffs, baseline,width, height, ndisp, isint, vmin, vmax,dyavg, dymax = 114.291, 174.019, 2960, 2016, 250,0, 38, 222, 0,0
        params = (doffs, baseline,width, height, ndisp, isint, vmin, vmax,dyavg, dymax)
        
    images = []
    for f in sorted(glob.glob(DataPath+"*.png")):
        images.append(rgb(cv2.imread(f)))
    
    return images, K1, K2, params

def preprocess(impaths):
    im1 = UndistortImage(bayer2rgb(cv2.imread(impaths[i],0)), LUT)[:800,:,:]
    im2 = UndistortImage(bayer2rgb(cv2.imread(impaths[i+1],0)), LUT)[:800,:,:]
    return im1,im2

def bayer2rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_BAYER_GR2RGB)

def rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def gray(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

def drawlines(im1,im2,lines,pts1,pts2):
    '''
    reference: https://answers.opencv.org/question/38682/strange-behavior-of-findfundamentalmat-ransac/
    
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines
    '''
    lines = lines.reshape(-1,3)
    img1 = im1.copy()
    img2 = im2.copy()

    r,c = img1.shape[:2]
#     img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#     img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(100,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(np.int32(pt1)),4,color,-1)
        img2 = cv2.circle(img2,tuple(np.int32(pt2)),4,color,-1)
    return img1,img2


def SIFTpoints(im1,im2):
    MIN_MATCH_COUNT = 10
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1,None)
    kp2, des2 = sift.detectAndCompute(im2,None)


    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0,0] for i in range(len(matches))]

    # store all the good matches as per Lowe's ratio test.
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)
            matchesMask[i]=[1,0]
    
    draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0)
    
    im_out = cv2.drawMatchesKnn(im1,kp1,im2,kp2,matches,None,**draw_params)
    pts1 = np.array([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    
    return pts1,pts2, im_out

def epiLines(pts1_, whichImage, F, im_width):
    lines = cv2.computeCorrespondEpilines(pts1_.reshape(-1,1,2), whichImage,F).reshape(-1,3)
    lineOut = []
    for r in lines:
        x00,y00 = map(int, [0, -r[2]/r[1] ])
        x01,y01 = map(int, [im_width, -(r[2]+r[0]*im_width)/r[1] ])
        epiline = [[x00,y00], [x01,y01]]
        lineOut.append(epiline)        
    return np.array(lineOut)

def warpEpilines(lines1_, H1):

    el_start = np.float32(lines1_[:,0].reshape(-1,1,2))
    el_end = np.float32(lines1_[:,1].reshape(-1,1,2))

    el_start_ = cv2.perspectiveTransform(el_start, H1).squeeze()
    el_end_ = cv2.perspectiveTransform(el_end, H1).squeeze()

    lines = []
    for start, end in zip(el_start_, el_end_):
        lines.append([start, end])

    lines = np.array(lines)

    return lines

def ProjectionMatrix(K,R,C):
    I  = np.identity(3)
    return np.dot(K, np.dot(R, np.hstack((I, -C))))

def projectPts(R, C, x3D, K):
    I  = np.identity(3)
    P2 = np.dot(K, np.dot(R, np.hstack((I, -C.reshape(3,1)))))
    x3D_4 = homo(x3D)
    x_proj = np.dot(P2, x3D_4.T)
    x_proj = (x_proj/x_proj[2,:]).T
    return x_proj

def homo(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))


def rescale(img, scale_percent = 60):

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def foldercheck(Savepath):
    if(not (os.path.isdir(Savepath))):
        print(Savepath, "  was not present, creating the folder...")
        os.makedirs(Savepath)