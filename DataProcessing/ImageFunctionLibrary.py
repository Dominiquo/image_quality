import cv2
import numpy as np
from numpy.linalg import norm
import cv2
import glob


def rescale_image(rescale_factor=(1./255)):
	return lambda im: im*rescale_factor


def canny_edge_avg_pixels(img, min_val=100, max_val=200):
	h,w,chan = img.shape
	edges = cv2.Canny(np.uint2(img), min_val, max_val)*(1.0/255)
	edge_pixel_count = sum(sum(edges))
	avg_edge_count = edge_pixel_count/(h*w)
	return avg_edge_count


def extract_histogram(image, cmask):
    im = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[0] 
    channels = [0]
    histSize = [60]
    histRange = [0,250]
    hist_item = cv2.calcHist([im],channels,cmask,histSize,histRange)
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist = hist_item * (1.0 / np.sum(hist_item))
    return hist


def extract_hog(img):
    # Extracts a 64 dim array for the image
    samples = []
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin_n = 16
    bin = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= norm(hist) + eps

    samples.append(hist)
    return np.float32(samples)


