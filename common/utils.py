"""
Common utilities for landmark detection.

Provides bounding box handling, landmark visualization, and image processing utilities.
"""

import cv2
import numpy as np


class BBox(object):
    """
    Bounding box representation for face detection.
    
    Args:
        bbox: List of [left, right, top, bottom] coordinates
    """
    def __init__(self, bbox):
        self.left = bbox[0]
        self.right = bbox[1]
        self.top = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]

    def projectLandmark(self, landmark):
        """
        Project landmarks from absolute coordinates to normalized [0, 1] range.
        
        Args:
            landmark: Array of (x, y) coordinates in absolute pixel space
            
        Returns:
            Normalized landmarks in [0, 1] range
        """
        landmark_= np.asarray(np.zeros(landmark.shape))     
        for i, point in enumerate(landmark):
            landmark_[i] = ((point[0]-self.x)/self.w, (point[1]-self.y)/self.h)
        return landmark_

    def reprojectLandmark(self, landmark):
        """
        Reproject landmarks from normalized [0, 1] range to absolute coordinates.
        
        Args:
            landmark: Array of (x, y) coordinates in normalized [0, 1] range
            
        Returns:
            Landmarks in absolute pixel coordinates
        """
        landmark_= np.asarray(np.zeros(landmark.shape)) 
        for i, point in enumerate(landmark):
            x = point[0] * self.w + self.x
            y = point[1] * self.h + self.y
            landmark_[i] = (x, y)
        return landmark_

def drawLandmark(img, bbox, landmark):
    """
    Draw landmarks and bounding box on an image.
    
    Args:
        img: Input image (grayscale or RGB)
        bbox: BBox object representing the face bounding box
        landmark: Array of (x, y) landmark coordinates
        
    Returns:
        Image with landmarks and bounding box drawn
    """
    img_ = img.copy()
    cv2.rectangle(img_, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img_, (int(x), int(y)), 3, (0,255,0), -1)
    return img_

def drawLandmark_multiple(img, bbox, landmark):
    """
    Draw landmarks and bounding box on an image (in-place modification).
    
    Args:
        img: Input image (grayscale or RGB), modified in-place
        bbox: BBox object representing the face bounding box
        landmark: Array of (x, y) landmark coordinates
        
    Returns:
        Modified image with landmarks and bounding box drawn
    """
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
    return img

def drawLandmark_Attribute(img, bbox, landmark, gender, age):
    """
    Draw landmarks, bounding box, and attributes (gender, age) on an image.
    
    Args:
        img: Input image (grayscale or RGB)
        bbox: BBox object representing the face bounding box
        landmark: Array of (x, y) landmark coordinates
        gender: Gender prediction array
        age: Age prediction array
        
    Returns:
        Image with landmarks, bounding box, and attributes drawn
    """
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 3, (0,255,0), -1)
        if gender.argmax()==0:
                # -1->female, 1->male; -1->old, 1->young
                cv2.putText(img, 'female', (int(bbox.left), int(bbox.top)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        else:
                cv2.putText(img, 'male', (int(bbox.left), int(bbox.top)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),3)
        if age.argmax()==0:
                cv2.putText(img, 'old', (int(bbox.right), int(bbox.bottom)),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), 3)
        else:
                cv2.putText(img, 'young', (int(bbox.right), int(bbox.bottom)),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), 3)
    return img


def drawLandmark_only(img, landmark):
    """
    Draw only landmarks on an image (without bounding box).
    
    Args:
        img: Input image (grayscale or RGB)
        landmark: Array of (x, y) landmark coordinates
        
    Returns:
        Image with landmarks drawn
    """
    img_=img.copy()
    #cv2.rectangle(img_, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img_, (int(x), int(y)), 3, (0,255,0), -1)
    return img_


def processImage(imgs):
    """
    Normalize images by subtracting mean and dividing by standard deviation.
    
    Args:
        imgs: Array of images with shape [N, 1, W, H]
        
    Returns:
        Normalized images
    """
    imgs = imgs.astype(np.float32)
    for i, img in enumerate(imgs):
        m = img.mean()
        s = img.std()
        imgs[i] = (img-m)/s
    return imgs

def flip(face, landmark):
    """
    Horizontally flip a face image and adjust landmark coordinates accordingly.
    
    Args:
        face: Face image to flip
        landmark: Array of (x, y) landmark coordinates
        
    Returns:
        Tuple of (flipped_face, flipped_landmarks)
    """
    face_ = cv2.flip(face, 1) # 1 means flip horizontal
    landmark_flip = np.asarray(np.zeros(landmark.shape))
    for i, point in enumerate(landmark):
        landmark_flip[i] = (1-point[0], point[1])
    # for 5-point flip
        landmark_flip[[0,1]] = landmark_flip[[1,0]]
    landmark_flip[[3,4]] = landmark_flip[[4,3]]
    # for 19-point flip
        #landmark_flip[[0,9]] = landmark_flip[[9,0]]
    #landmark_flip[[1,8]] = landmark_flip[[8,1]]
    #landmark_flip[[2,7]] = landmark_flip[[7,2]]
    #landmark_flip[[3,6]] = landmark_flip[[6,3]]
    #landmark_flip[[4,11]] = landmark_flip[[11,4]]
    #landmark_flip[[5,10]] = landmark_flip[[10,5]]
    #landmark_flip[[12,14]] = landmark_flip[[14,12]]
    #landmark_flip[[15,17]] = landmark_flip[[17,15]]
    return (face_, landmark_flip)

def scale(landmark):
    """
    Scale landmarks from [0, 1] range to [-1, 1] range.
    
    Args:
        landmark: Array of normalized landmarks in [0, 1] range
        
    Returns:
        Landmarks scaled to [-1, 1] range
    """
    landmark_ = np.asarray(np.zeros(landmark.shape))
    lanmark_=(landmark-0.5)*2
    return landmark_

def check_bbox(img, bbox):
    """
    Check if bounding box is within image boundaries.
    
    Args:
        img: Input image
        bbox: BBox object to check
        
    Returns:
        True if bbox is within image bounds, False otherwise
    """
    img_w, img_h = img.shape
    if bbox.x > 0 and bbox.y > 0 and bbox.right < img_w and bbox.bottom < img_h:
        return True
    else:
        return False

def rotate(img, bbox, landmark, alpha):
    """
    Rotate an image and adjust bounding box and landmarks accordingly.
    
    Args:
        img: Input image
        bbox: BBox object representing the face bounding box
        landmark: Array of (x, y) landmark coordinates
        alpha: Rotation angle in degrees
        
    Returns:
        Tuple of (rotated_face, rotated_landmarks) in absolute coordinates
    """
    center = ((bbox.left+bbox.right)/2, (bbox.top+bbox.bottom)/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, img.shape)
    landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
    face = img_rotated_by_alpha[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
    return (face, landmark_)
