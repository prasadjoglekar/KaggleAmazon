import numpy as np # linear algebra
import os
from skimage import io, exposure, color
from skimage import feature
import cv2
from matplotlib import pyplot as plt
from skimage.feature._hog import hog


def computeHistogramJPG(image_path):
   
    numBins= [16]
    img = cv2.imread(image_path)
    #img = cv2.imread('home.jpg',0)
    color = ('b','g','r')
    histArr = []
    for i in enumerate(color):
        histr = cv2.calcHist([img],[i],None,numBins,[0,256])
        histArr.extend(histr.ravel().tolist())
    return histArr
    

#bag of colors model:
def computeBagOfColorsTIF(image_path):
    #for each image in the dataframe, read the rgba values, then compute average and variance of rgba, them and add back to the df.
    rgbn_image = io.imread(image_path)
    
    r, g, b, a = rgbn_image[:, :, 0], rgbn_image[:, :, 1], rgbn_image[:, :, 2], rgbn_image[:, :, 3]
    avg_r = np.mean(r)
    avg_g = np.mean(g)
    avg_b = np.mean(b)
    avg_a = np.mean(a)
    var_r = np.var(r)
    var_g = np.var(g)
    var_b = np.var(b)
    var_a = np.var(a)
    
    avg_features = [avg_r, avg_g, avg_b, avg_a, var_r, var_g, var_b, var_a]
    return avg_features    
    
def computeBagOfColorsJPG(image_path):
    numBins=16

    rgb_image = io.imread(image_path)
    
    rr, gg, bb  = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    avg_r = np.mean(rr)
    avg_g = np.mean(gg)
    avg_b = np.mean(bb)
    var_r = np.var(rr)
    var_g = np.var(gg)
    var_b = np.var(bb)
    
    hist_r = np.histogram(rr, bins=numBins)[0]
    hist_g = np.histogram(gg, bins=numBins)[0]
    hist_b = np.histogram(bb, bins=numBins)[0]
    
    avg_features = [avg_r, avg_g, avg_b, var_r, var_g, var_b]
    avg_features.extend(hist_r)
    avg_features.extend(hist_g)
    avg_features.extend(hist_b)
    return avg_features

def computeHOG(image_path):
    rgb_image = io.imread(image_path)
    gray_image = color.rgb2gray(rgb_image)
    fd = hog(gray_image, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(3, 3), visualise=False)
    
    return fd
    

def computeEdgeCountJPG(image_path):
    
    rgb_image = io.imread(image_path)
    rr, gg, bb  = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    sigma=1.2
    rr_e = feature.canny(rr, sigma = sigma)
    gg_e = feature.canny(gg, sigma = sigma)
    bb_e = feature.canny(bb, sigma = sigma)
    edgeCount = [np.count_nonzero(rr_e), np.count_nonzero(gg_e), np.count_nonzero(bb_e)]
    return edgeCount

    
if __name__ == "__main__":
    os.chdir("C:/Users/Prasad/git/KaggleAmazon/Amazon/src")

    #computeHistogramJPG("C:/Users/Prasad/git/KaggleAmazon/Amazon/input/train-jpg-sample/train_10066.jpg")
    computeHOG("C:/Users/Prasad/git/KaggleAmazon/Amazon/input/train-jpg-sample/train_10082.jpg")