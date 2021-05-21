import numpy as np

# some parts of the code could be similar from below link, since I am a co-author of the below code base for a project in CMSC733 
#  https://github.com/sakshikakde/Buildings-built-in-minutes-An-SfM-Approach,



def normalize(uv):

    uv_dash = np.mean(uv, axis=0)
    u_dash ,v_dash = uv_dash[0], uv_dash[1]

    u_cap = uv[:,0] - u_dash
    v_cap = uv[:,1] - v_dash

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_dash],[0,1,-v_dash],[0,0,1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(x_.T)).T

    return  x_norm, T

def FundamentalMatrix(pt1,pt2):

    if pt1.shape[0] > 7:
        

        pt1_n, T1 = normalize(pt1)
        pt2_n, T2 = normalize(pt2)
            
        A = np.zeros((len(pt1_n),9))
        for i in range(0, len(pt1_n)):
            x_1,y_1 = pt1_norm[i][0], pt1_norm[i][1]
            x_2,y_2 = pt2_norm[i][0], pt2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        U, S, Vt = np.linalg.svd(A, full_matrices=True)
        F = Vt.T[:, -1]
        F = F.reshape(3,3)

        u, s, vt = np.linalg.svd(F)
        s = np.diag(s)
        s[2,2] = 0
        F = np.dot(u, np.dot(s, vt))

        F = np.dot(T2.T, np.dot(F, T1))
        return F

    else:
        return None

def EpipolarConstraintError(pt1, pt2, F): 

    pt1tmp=np.array([pt1[0], pt1[1], 1]).T
    pt2tmp=np.array([pt2[0], pt2[1], 1])

    error = pt1tmp.dot(F.dot(pt1_tmp))
    
    return np.abs(error)


def RANSAC(pts1, pts2, error_thresh = 0.02, n_iterations = 1000):
    
    
    max_inliers = 0
    chosen_indices = []
    chosen_f = 0
    n_rows = pts1.shape[0]
    for i in range(0, n_iterations):
        indices = []
        #select 8 points randomly
        
        random_indices = np.random.choice(n_rows, size=8)
        pt1_8 = pts1[random_indices]
        pt2_8 = pts2[random_indices]
        
        F = FundamentalMatrix(pt1_8, pt2_8)
        for j in range(n_rows):
            error = EpipolarConstraintError(pts1[j], pts2[j], F)
            if error < error_thresh:
                indices.append(j)

        if len(indices) > max_inliers:
            max_inliers = len(indices)
            inliers = indices
            chosen_f = F

    pts1_inliers, pts2_inliers = pts1[inliers], pts2[inliers] 
    return F, pts1_inliers, pts2_inliers

def getEssentialMatrix(K1, K2, F):
    E = K2.T.dot(F).dot(K1)
    U,s,V = np.linalg.svd(E)
    s = [1,1,0]
    E_corrected = np.dot(U,np.dot(np.diag(s),V))
    return E_corrected