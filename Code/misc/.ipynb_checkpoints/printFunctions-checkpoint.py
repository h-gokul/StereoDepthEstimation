import cv2
import numpy as np
import matplotlib.pyplot as plt


def drawEpilines(im1,im2,lines,pts1,pts2):
    img1,img2 = im1.copy(),im2.copy()
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    lines = lines.reshape(-1,3)
    r,c = img1.shape[:2]
    for r,pt1,pt2 in zip(lines,np.int32(pts1),np.int32(pts2)):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),2,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),2,color,-1)
    return img1,img2

def drawLines(im1, lines, pts):
    img1 = im1.copy()
    for l, pt in zip(lines,pts):
        l_color = (0,255,0)
        (x0,y0), (x1,y1) = l[0], l[1]
        img1 = cv2.line(img1, (x0,y0), (x1,y1), l_color,1)
        
        p_color = (0,0,255)
        x, y = pt[0], pt[1]
        cv2.circle(img1, (int(x), int(y)), 1, p_color, -1)
        
    return img1

def drawProjectedPoints(im, pts2_, x_proj):
    ## blue ground truth, 
    ## green reprojected points
    im2 = im.copy()
    x = pts2_[:,0]
    y = pts2_[:,1]
    x_ = x_proj[:,0]
    y_ = x_proj[:,1]

    for i in range(x.shape[0]):

        x1, y1 = x[i], y[i]
        x2, y2 = x_[i], y_[i]
        cv2.circle(im2, (int(x1), int(y1)), 3, (0,0,255), -1)
        cv2.circle(im2, (int(x2), int(y2)), 3, (0,255,0), -1)
    return im2
