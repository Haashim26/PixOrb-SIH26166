"""SIH26166 Stage 4: RANSAC affine/homography verification."""
import cv2
import numpy as np

def ransac_verify(raw_matches, model="homography", threshold=3.0):
    if len(raw_matches)<4: raise ValueError("At least 4 candidate correspondences are required")
    src=np.float32([[m[0],m[1]] for m in raw_matches]); ref=np.float32([[m[2],m[3]] for m in raw_matches])
    if model=="affine":
        M,mask=cv2.estimateAffinePartial2D(src,ref,method=cv2.RANSAC,ransacReprojThreshold=threshold,maxIters=5000,confidence=0.999)
        if M is None: raise RuntimeError("RANSAC affine estimation failed")
        H=np.vstack([M,[0,0,1]]).astype(np.float64)
    else:
        H,mask=cv2.findHomography(src,ref,cv2.RANSAC,threshold,maxIters=10000,confidence=0.999)
        if H is None: raise RuntimeError("RANSAC homography estimation failed")
    mask=mask.ravel().astype(bool)
    inliers=[raw_matches[i] for i in range(len(raw_matches)) if mask[i]]
    return H.astype(np.float64),inliers,mask

def project(H,pts):
    p=np.c_[np.asarray(pts,float),np.ones(len(pts))]
    q=(H@p.T).T; return q[:,:2]/q[:,2:3]

def residuals(H,matches):
    if not matches:return np.empty(0)
    src=np.array([[m[0],m[1]] for m in matches],float); ref=np.array([[m[2],m[3]] for m in matches],float)
    return np.linalg.norm(project(H,src)-ref,axis=1)
