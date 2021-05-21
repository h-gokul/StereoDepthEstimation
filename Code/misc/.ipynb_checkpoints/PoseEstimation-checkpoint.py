import numpy as np
from misc.utils import *


##########################################################
############## Linear Triangulation Block ################
##########################################################
def A_crossProd(P1, P2, pt1, pt2):
    p1, p2, p3 = P1
    p1_, p2_, p3_ = P2

    p1, p2, p3 =  p1.reshape(1,-1), p2.reshape(1,-1), p3.reshape(1,-1) # shape 1x4
    p1_, p2_, p3_ =  p1_.reshape(1,-1), p2_.reshape(1,-1), p3_.reshape(1,-1) # shape 1x4

    x, y = pt1
    x_, y_ = pt2
    A = np.vstack((   y*p3 - p2,    p1-x*p3,
                   y_*p3_ - p2_, p1_-x_*p3_ ))
    return A
                     
def triangulate(pts1, pts2, C1, R1, C2, R2, K1, K2):
    
    P1 = ProjectionMatrix(K1,R1,C1)
    P2 = ProjectionMatrix(K2,R2,C2)


    X3D =[]
    for x1,x2 in zip(pts1, pts2):
        A = A_crossProd(P1, P2, x1, x2)
        _,_,Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X/X[-1]
        X3D.append(X[:3])
    return np.array(X3D)

def LinearTriangulation(r_set, c_set, x1, x2, K1, K2):
    x3D_set = []
#     x3D_set_ref = []
    for i in range(len(r_set)):
        R1, R2 = np.identity(3), r_set[i] # 3x3
        C1, C2 = np.zeros((3, 1)),  c_set[i].reshape(3,1) # 3x1
        x3D = triangulate(np.float32(x1), np.float32(x2), C1, R1, C2, R2, K1, K2)    
        x3D_set.append(x3D)

#         x3D, _, _ = triangulate_cv2(np.float32(x1), np.float32(x2), C1, R1, C2, R2, K)    
#         x3D_set_ref.append(x3D)    
#     return x3D_set, x3D_set_ref
    return x3D_set

##########################################################
##################### Pose Estimation ####################
##########################################################



def decomposeEssentialMat(E):
    """
    Given essential matrix E and Intrinsic parameters K, dec
    Args:
        E (array): Essential Matrix
        K (array): Intrinsic Matrix
    Returns:
        arrays: set of Rotation and Camera Centers
    """

    ##UPDATE
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # print("E svd U", U)
    # print("E svd S", S)
    # print("E svd U[:, 2]", U[:, 2])
    R_set = []
    C_set = []
    R_set.append(np.dot(U, np.dot(W, V_T)))
    R_set.append(np.dot(U, np.dot(W, V_T)))
    R_set.append(np.dot(U, np.dot(W.T, V_T)))
    R_set.append(np.dot(U, np.dot(W.T, V_T)))
    C_set.append(U[:, 2])
    C_set.append(-U[:, 2])
    C_set.append(U[:, 2])
    C_set.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(R_set[i]) < 0):
            R_set[i] = -R_set[i]
            C_set[i] = -C_set[i]

    return R_set, C_set

def recoverPose(E, x1, x2, K1, K2):
    
    r_set, c_set = decomposeEssentialMat(E)
    x3D_set = LinearTriangulation(r_set, c_set, x1, x2, K1, K2)
    
    best_i = 0
    max_positive_depths = 0
    
    for i in range(len(r_set)):
        R, C = r_set[i],  c_set[i].reshape(-1,1) 
        r3 = R[2].reshape(1,-1)
        x3D = x3D_set[i]
        n_positive_depths = DepthPositivityConstraint(x3D, r3,C)
        if n_positive_depths > max_positive_depths:
            best_i = i
            max_positive_depths = n_positive_depths 
#         print(n_positive_depths, i, best_i)

    R, C, x3D = r_set[best_i], c_set[best_i], x3D_set[best_i]

    return R, C, x3D 

def DepthPositivityConstraint(x3D, r3, C):
    # r3(X-C) check positivity in Camera 2. z = X[2] check positivity in Camera 1 at reference 
    n_positive_depths=  0
    for X in x3D:
        X = X.reshape(-1,1) 
        if r3.dot(X-C)>0 and X[2]>0: 
            n_positive_depths+=1
    return n_positive_depths