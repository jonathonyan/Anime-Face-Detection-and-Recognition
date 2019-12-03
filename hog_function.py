import numpy as np
import cv2
import pdb

def my_hog_function(Img, resize_size = (64, 64), cell_size=(8, 8), bin_size=9):
    I = cv2.resize(Img, resize_size)
    dx =  cv2.Sobel(I,cv2.CV_64F,1,0,ksize=1)
    dy =  cv2.Sobel(I,cv2.CV_64F,0,1,ksize=1)

    magnitudes = np.sqrt(dx**2 + dy ** 2)

    angles = cv2.phase(dx, dy)

    histograms = []

    for i in range(0, I.shape[0], cell_size[1]):
        for j in range(0, I.shape[1], cell_size[0]):
        
            m_each = magnitudes[i:i+cell_size[1], j:j+cell_size[0]].flatten()
            a_each = angles[i:i+cell_size[1], j:j+cell_size[0]].flatten()
            max_each = max(np.max(m_each), np.max(a_each))
            min_each = min(np.min(m_each), np.min(a_each))
            histogram_each = []
            diff_each = max_each - min_each

            diff_each /= bin_size

            for k in range(bin_size):
                down = k * diff_each
                up = (k+1) * diff_each
                count = 0
                count += len(m_each[(m_each>= down) & (m_each < up) ])
                count += len(a_each[(a_each>= down) & (a_each < up) ])

                if k == bin_size - 1:
                    count = 0
                    count += len(m_each[(m_each>= down) & (m_each <= up)])
                    count += len(a_each[(a_each>= down) & (a_each <= up) ])


                histogram_each.append(count)
            
            histograms.append(histogram_each)


    for hist in histograms:
        hist = hist / np.linalg.norm(hist)
    
    return np.array(histograms).flatten()
    


