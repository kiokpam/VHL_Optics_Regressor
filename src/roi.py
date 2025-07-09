import cv2
import numpy as np
from matplotlib import pyplot as plt

from squares import find_squares
from config import HSV_KITS, RATIO, SIZE_IMG, SIZE_CUT


def cutImage(image, ratio=None, size=None):
    """Crop the image to the specified ratio and size.

    Args:
        image: The input image
        ratio: The aspect ratio for cropping
        size: The size of the cropped image

    return: The cropped image
    """
    # Set default values for ratio and size if not provided
    if ratio is None:
        ratio = RATIO
    if size is None:
        size = SIZE_CUT
    # Check if the image is empty 
    if image is None:
        raise ValueError("The image is empty. Please check the input image path.")

    # Get the coordinates of middle of the image
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2

    # Cropping the image with ratio 3:4 for the center of the image
    # Calculate the cropping box dimensions
    crop_width = int(width * ratio)  # 3:4 ratio
    crop_height = int(height * ratio)  # 3:4 ratio
    x1 = max(center_x - crop_width // 2, 0)
    y1 = max(center_y - crop_height // 2, 0)
    x2 = min(center_x + crop_width // 2, width)
    y2 = min(center_y + crop_height // 2, height)
    # Crop the image using the calculated coordinates
    cropped_image = image[y1:y2, x1:x2]
    # Resize the cropped image to 300x300 pixels
    resized_image = cv2.resize(cropped_image, size, interpolation=cv2.INTER_AREA)
    return resized_image


def resizeImage(image, size=None):
    """Crop the image to a square centered around the middle of the image.

    Args:
        image: The input image
        size: The size of the square crop

    return: The cropped square image
    """
    # Set default size if not provided
    if size is None:
        size = SIZE_IMG
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the center of the image
    center_x = width // 2
    center_y = height // 2

    # Calculate the cropping box dimensions
    crop_size = size  # 224x224 square crop
    x1 = max(center_x - crop_size // 2, 0)
    y1 = max(center_y - crop_size // 2, 0)
    x2 = min(center_x + crop_size // 2, width)
    y2 = min(center_y + crop_size // 2, height)

    # Crop the image using the calculated coordinates
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image


def getSquaredImage(image, kit=None, size=None):
    """Get the squared frame of the image.

    Args:
        cropped_image: The cropped image
        kit: The HSV kit for color detection
        size: The size of the squared frame
        
    :return: The squared frame of the image
    """
    # Set default kit and size if not provided
    if kit is None:
        kit = HSV_KITS['1.1.1.1.0']
    if size is None:
        size = SIZE_IMG
    
    # Change the color space to HSV
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask using the HSV kit
    mask = cv2.inRange(img_hsv, kit[0], kit[1])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if not contours:
        # raise ValueError("Your image does not have any countours")
        return None
    
    x, y, w, h = cv2.boundingRect(contours[0])
    roi = image[y : y + h, x : x + w]
    # Resize the cropped image to the specified size
    roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)

    return roi


def getSample(image):
    # get the mask of the image
    squares = find_squares(image)
    x, y, w, h = cv2.boundingRect(squares[(len(squares)//2)])
    sample = image[y:y+h, x:x+w]
    # Create a mask with the size x, y, w, h
    roi = np.zeros(image.shape[:2], dtype=np.uint8)
    roi[y:y+h, x:x+w] = 255
    # Reverse the mask
    mask = cv2.bitwise_not(roi)
    background = cv2.bitwise_and(image, image, mask=mask)
    return sample, background, roi



def runROI(
    image, 
    ratio:float= None,
    size_cut:int= None,
    size_img:int= None,
    kit=None, 
    ):
    """Run the ROI process on the image.

    Args:
        image: The input image (BGR format)
        ratio: The aspect ratio for cropping
        size_cut: The size of the cropped image
        size_img: The size of the squared frame
        kit: The HSV kit for color detection
    """
    if kit is None:
        kit = HSV_KITS['1.1.1.1.0']

    if size_cut is None:
        size_cut = SIZE_CUT

    if size_img is None:
        size_img = SIZE_IMG

    if ratio is None:
        ratio = RATIO
    try:
        # Crop and resize the image
        cropped_image = cutImage(image=image, ratio=ratio, size=size_cut)  #3/4
        resize_image = resizeImage(image=cropped_image, size=size_img)

        # Get the squared image using the specified HSV kit and size
        squared_frame = getSquaredImage(image=resize_image, kit=kit, size=size_img)
        sample, background, roi = getSample(image=squared_frame)

        return squared_frame, sample, background, roi
    
    except:
        return None


if __name__ == "__main__":
    # Example usage
    root = "D:/Work/VHL/VHL_Optics/data/dataset/oppo/coomassie blue/"
    image_path = "rls8Q_ngochanpham274@gmail.com_2025-03-31 15_22_55_Tien_oppo_Coomassie Blue_30ppm_2_6__10.8769248_106.6780862.jpg"
    image = cv2.imread(root+image_path)
    # cropped_image = cutImage(image, ratio=RATIO, size=SIZE_CUT)
    # squared_frame = resizeImage(cropped_image) 
    # roi_img = getSquaredImage(squared_frame, kit=HSV_KITS['1.1.1.1.0'])
    squared_frame, sample, background, roi = runROI(image)

    # Display the images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 5, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 5, 2)
    plt.imshow(cv2.cvtColor(squared_frame, cv2.COLOR_BGR2RGB))
    plt.title("Squared Image")

    plt.subplot(1, 5, 3)
    plt.imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
    plt.title("Sample")

    plt.subplot(1, 5, 4)
    plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    plt.title("Background")

    plt.subplot(1, 5, 5)
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.title("ROI")
    plt.show()
