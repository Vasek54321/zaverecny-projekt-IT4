import cv2
import numpy as np
from matplotlib import pyplot as plt

def car_detection(image1, image2, o):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    # Load images
    if o == 1:
        img = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    else: 
        img = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    fgmask = fgbg.apply(img)
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Najít největší konturu
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000:  # Odfiltrování šumu
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center_x = x + w // 2
            center_y = y + h // 2
            current_position = (center_x, center_y)
    
            # Create a blank mask
            mask = np.zeros_like(img)

            # Fill the box area in the mask with white (255, 255, 255)
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), thickness=-1)

            # Apply the mask to the image
            result = cv2.bitwise_and(img, mask)

            # Save or display the result
            if o == 1:
                cv2.imwrite('image1_adj.png', result)
            else:
                cv2.imwrite('image2_adj.png', result)

def compare_images(image1_adj, image2_adj):
    # Load images
    img1 = cv2.imread(image1_adj, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_adj, cv2.IMREAD_GRAYSCALE)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw all matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matches
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title('SIFT Feature Matching')
    plt.show()

if __name__ == "__main__":
    image1 = 'image-1.png'
    image2 = 'image-2.png'

    o = 1
    car_detection(image1, image2, o)
    o = 2
    car_detection(image1, image2, o)

    image1_adj = 'image1_adj.png'
    image2_adj = 'image2_adj.png'
    compare_images(image1_adj, image2_adj)