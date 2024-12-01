
import numpy as np
import argparse
import cv2
def rectCountour(countours):
    rectCon=[]
    for i in countours:
        
        area=cv2.contourArea(i)
        #print(area)
        if area>60000:
         
          peri=cv2.arcLength(i,True)
          approx=cv2.approxPolyDP(i,0.02*peri,True)
          #print("contors points",approx)
          #if len(approx)==4:
          rectCon.append(i)
    rectCon=sorted(rectCon,key=cv2.contourArea,reverse=True) 
    
    return rectCon
# .... Code for getting the all corners point
def getCornesPts(big_con):
   peri=cv2.arcLength(big_con,True)
   approx=cv2.approxPolyDP(big_con,0.02*peri,True)
   return approx
   
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

def splitboxes(img):
    height, width = img.shape[:2]
    

# Number of splits
    num_splits = 25
    w_n_split=5

# Calculate the required padding
    cropped_height = (height // num_splits) * num_splits
    cropped_width = (width // w_n_split) * w_n_split
    top_crop=height-cropped_height
    top_crop_w=width-cropped_width
    
    padded_image =img[top_crop:height, :] 
    rows= np.vsplit(padded_image,25)
    
    
   
    boxes=[]
    i=0
    for r in rows:
        r_p =r[:,top_crop_w:width] 
        i=0
        cols=np.hsplit(r_p,5)
        for box in cols:
            if i==0:
                i=i+1
            else:
                boxes.append(box)
                i=i+1
 
    return boxes  

def allpixelValue(questions,choice):
    pixelVal=np.zeros((questions,choice))
    countC=0
    countR=0
    i=1
    for image in boxes:
       total_pixel=cv2.countNonZero(image)
       pixelVal[countR][countC]=total_pixel
       if (countC==choice-1):
         countR+=1
         countC=0
       else:
         countC+=1
   
    return pixelVal


             



image = cv2.imread("saved_image/scanned_document.jpg")
#image=cv2.imread("images\omr_6_qn.webp")
original = image.copy()
image_con= image.copy()
doc_corners=""
questions=25
choice=4
ans={ 1: 1, 2: 2, 3: 2, 4: 3,5:4,6:3,7:' ',8:' ',9:2,10:3,11:4,12:3,13:' ',14:1,15:' ',16:3,17:' ',18:' ',19:2,20:' ',21:' ',22:' ',23:1,24:' ',25:' '}

    # Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3,3), 0)
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)
#edged = cv2.Canny(blurred, 75, 200)
_, thresh = cv2.threshold(sharpened, 190, 255, cv2.THRESH_BINARY_INV)
#thresh = cv2.threshold(sharpened, 0, 255,
 #                      	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#cv2.imshow("thresh2",thresh)
edges = cv2.Canny(thresh, 50, 150)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cont_img=cv2.drawContours(original, contours, -1, 255, 3)
rectCon=rectCountour(contours)

big_con=rectCon[0]
doc_corners=getCornesPts(big_con)


if big_con.size!=0:
   cv2.drawContours(image_con,big_con,-1, 255, 3)
   ordered_points = order_points(doc_corners.reshape(4, 2))
   (tl, tr, br, bl) = ordered_points
   
   width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
   height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
   dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
   ], dtype="float32")
   matrix = cv2.getPerspectiveTransform(ordered_points, dst)
   ques_25 = cv2.warpPerspective(original, matrix, (width, height))
   
gray = cv2.cvtColor(ques_25, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)
_, thresh_q_25 = cv2.threshold(sharpened, 140, 255, cv2.THRESH_BINARY_INV)
height, width = thresh_q_25.shape[:2] 

boxes=splitboxes(thresh_q_25)
pixelVal=allpixelValue(questions,choice)

# finding index value of each correct ans
index=[]
for x in range(0,questions):
    each_q=pixelVal[x]
    index_val=np.where(each_q==np.amax(each_q))
    
    index.append(index_val[0][0])

total=0
correct=0
# ....mapping the index with answer key.....
for i in range (len(index)):
    if ans[i+1]!=' ':
        total+=1
        if ans[i+1]==index[i]+1:
            correct+=1

score=(correct/total)*100
print("score: ",score,"%")





#cv2.imshow("thresh_q_25",thresh_q_25)
#cv2.imshow("thresh",gray)
#cv2.imshow("edges",edges)
cv2.waitKey(0)