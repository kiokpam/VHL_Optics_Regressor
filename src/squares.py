import numpy as np
import cv2
from matplotlib import pyplot as plt

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    try:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        squares = []
        for gray in cv2.split(img):
            for thrs in range(0, 255, 40):
                if thrs == 0:
                    bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                    bin = cv2.dilate(bin, None)
                else:
                    retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt_len = cv2.arcLength(cnt, True)
                    cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                    if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                        if max_cos < 0.3:
                            squares.append(cnt)

        squares = sorted(squares,key= cv2.contourArea, reverse=True)
        mean_area = []
        for square in squares:
            area = cv2.contourArea(square)
            if not area > 20000:  # Filter out large areas
                mean_area.append(square)
        mean_area = sorted(mean_area, key=cv2.contourArea, reverse=True)

        return mean_area
    except:
        return None

if __name__ == '__main__':
    from roi import cutImage, resizeImage, getSquaredImage
    # Use os.path for more robust path handling
    fn = './data/hinhanh/dataset/oppo/coomassie blue/'
    path = 'rls8Q_ngochanpham274@gmail.com_2025-03-31 15_22_55_Tien_oppo_Coomassie Blue_30ppm_2_6__10.8769248_106.6780862.jpg'
    img = cv2.imread(fn+path)
    if img is None:
        print(f"Error: Could not read image file: {fn}")
        exit(1)

    img = getSquaredImage(resizeImage(cutImage(img)))  
    squares = find_squares(img)
    # Get the mean area of lower_middle of the squares
    
    print(f'Number of squares found: {len(squares)}')
    cv2.drawContours(img, squares, 0, (0, 0, 255), 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Squares Detected')
    plt.axis('off')
    plt.show()
