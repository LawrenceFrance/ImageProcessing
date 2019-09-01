import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

pics = []
names = []

original1 = cv.imread("starfish.png")
#original2 = cv.cvtColor(original1, cv.COLOR_BGR2RGB)
# pics.append(original1)
# names.append('Original')


# GREY, BLACK AND WHITE

grey = cv.cvtColor(original1, cv.COLOR_BGR2GRAY)
# pics.append(grey)
# names.append('Greyscale')
#cv.imshow('greyscale',grey)


# BLURRING #######################################################################

# STANDARD

blurred = cv.medianBlur(original1,5)
# pics.append(blurred)
# names.append('Blurred')

# DENOISING

h       = 20
hColor  = 20

templateWindowSize  = 7
searchWindowSize    = 21
   
#pics.append(cv.cvtColor(cv.fastNlMeansDenoisingColored(original1, None,h,hColor,templateWindowSize,searchWindowSize),cv.COLOR_BGR2RGB))
denoise = cv.fastNlMeansDenoisingColored(original1, None,h,hColor,templateWindowSize,searchWindowSize)
# pics.append(denoise)
# names.append('Denoised')

# THRESHOLDING ###################################################################

# BINARY THRESHOLD

(thresh, bnw) = cv.threshold(grey, 127,255,cv.THRESH_BINARY)
# pics.append(bnw)
# names.append('Black n White')
#cv.imshow('black n white',bnw)

# compare with blurred

grey_blurred = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)
# pics.append(grey_blurred)
# names.append('Grey blurred')

(thresh, bnw_blurred) = cv.threshold(grey_blurred, 127,255,cv.THRESH_BINARY)
# pics.append(bnw_blurred)
# names.append('Black n white blurred')


# ADAPTIVE THRESHOLDING

# adap_mean  = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
# adap_gauss = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

# pics.append(adap_mean)
# names.append('Adaptive Mean')
# pics.append(adap_gauss)
# names.append('Adaptive Gaussian')


# EDGE DETECTION ####################################################################

# use denoised image, morphological edge detector
elKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13,13))
gradient = cv.morphologyEx(denoise, cv.MORPH_GRADIENT, elKernel)
pics.append(gradient)
names.append('Edges')

# canny edge detector
canny = cv.Canny(denoise,50,100)
pics.append(canny)
names.append('Canny')







for i,pic in enumerate(pics):
    cv.imshow(names[i],pic)


cv.waitKey(0)
cv.destroyAllWindows()



# DENOISING #######################################################################

