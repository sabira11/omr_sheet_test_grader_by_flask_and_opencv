import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_background(image_path):
    # Step 1: Read the image
    image = cv2.imread(image_path)
    original = image.copy()
    doc_corners=""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Edge detection
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

    #edges = cv2.Canny(sharpened, 40, 150)
    _, thresh = cv2.threshold(sharpened, 190, 255, cv2.THRESH_BINARY)
    #thresh = cv2.threshold(sharpened, 0, 129, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    #blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
    edges = cv2.Canny(thresh, 50, 150)
    edges = cv2.dilate(edges,None, iterations=1)
    
    #edges = cv2.erode(edges, None, iterations=1)
    # Step 3: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_img=cv2.drawContours(image, contours, -1, 255, 3)
    #cv2.imshow("cont_img",cont_img)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    
    # Assume the largest contour is the document
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:  # Quadrilateral found
            doc_corners = approx
            break
            
            #break
    else:
        print("Document contour not found.")
        return None
    
    def order_points(pts):
        # Rearrange contour points to top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        return rect

    ordered_points = order_points(doc_corners.reshape(4, 2))
    (tl, tr, br, bl) = ordered_points
    
    # Calculate the width and height of the document
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered_points, dst)
    warped = cv2.warpPerspective(original, matrix, (width, height))
    #cv2.imshow(" scanned_document", warped)
    
    # Step 6: Display the result
    #plt.figure(figsize=(10, 5))
    #plt.subplot(1, 3, 1)
    #plt.title("Original Image")
    #plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    
    #plt.subplot(1, 3, 2)
    #plt.title("Document Isolated")
    #plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    #plt.subplot(1, 3, 3)
    #plt.title("Document Isolated")
    #plt.imshow(cv2.cvtColor(cont_img, cv2.COLOR_BGR2RGB))
    #plt.show()
    
    return warped

# Example usage
output_image = remove_background("images/omr_sheet_2.jpg")

cv2.imwrite('saved_image/scanned_document.jpg',output_image)
