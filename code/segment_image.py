import cv2
from matplotlib.pyplot import axis
import numpy as np

def getBoundingBoxFromMask(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)




def segmented_image(image,prediction):
    
    image = image.astype(np.uint8)
    prediction = prediction.astype(np.uint8)
    
    img = cv2.resize(image,(720,1160),interpolation = cv2.INTER_NEAREST)
    pred = cv2.resize(prediction,(720,1160),interpolation = cv2.INTER_NEAREST)
    
    img = img.astype(np.uint8)
    pred = pred.astype(np.uint8)
    
    contours,_ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    approx = cv2.approxPolyDP(contours[0], 0.07 * cv2.arcLength(contours[0], True), True)
    box = np.squeeze(approx, axis=1)
    
    heigth = 70
    width = 180

    pt2 = np.float32([[0,0], [width,0], [0,heigth], [width,heigth]])

    if box[0][0] < box[2][0]:
        pt12 = np.float32([box[0],box[3],box[1],box[2]])
    else:
        pt12 = np.float32([box[1],box[0],box[2],box[3]])
        

    f = cv2.getPerspectiveTransform(pt12,pt2)
    f2 = cv2.warpPerspective(img,f,(width,heigth))


    gray = cv2.cvtColor(f2,cv2.COLOR_BGR2GRAY)
    # gray = f2[:,:,2]
    # gray = cv2.bilateralFilter(gray,11,17,17)
    canny = cv2.Canny(gray,40,200)
    # lines = cv2.HoughLinesP(canny,1,np.pi/180,10, maxLineGap=150)

    lines = cv2.HoughLinesP(canny,0.001,np.pi/180,3, minLineLength=40, maxLineGap=180)

    # if lines is None:
    #     break
    

    hough = np.zeros((heigth,width), np.uint8)
    hough = canny.copy()
    
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(hough,(x1,y1),(x2,y2),(255),1)
                

    kernel = np.ones((int(heigth * 0.1)), dtype=np.uint8) * 255

    opening = cv2.morphologyEx(hough, cv2.MORPH_OPEN, kernel)

    boxop = getBoundingBoxFromMask(opening).astype(int)
    
    result = f2[boxop[0]:boxop[2],boxop[1]:boxop[3]]

    if not result.any():
        result = f2.copy()
    resultSeg = result.copy()

    he, wi,dm = resultSeg.shape
    div = int(wi/7)

    divs = [i * div + 5 for i in range(1,7)]
    pad = 2
    divs[0] = divs[0] - pad
    divs[1] = divs[1] - pad
    divs[-2] = divs[-2] - pad
    divs[-1] = divs[-1] - pad


    for i in divs:
        resultSeg[:,i-1:i+1] = 0
        
    return [resultSeg,divs,f2,canny,opening]



def plate_characters(image,segModel,classModel):
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 
               'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X','Y', 'Z']
    # Calculate pred
    img = np.expand_dims(image,axis=0)
    pred = segModel.predict(img)
    
    pred = pred[0]
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0
    
    [resultSeg,divs,perspective,canny,opening] = segmented_image(img[0] * 255,pred)
    
    segImg = []
    for ind in range(0,len(divs)):
        if(ind == len(divs)-1):
            segImg.append(resultSeg[:,divs[ind]:,:])
        else:
            segImg.append(resultSeg[:,divs[ind]:divs[ind+1],:])
    
    resultMaps = []
    
    for letter in segImg:
        img = cv2.resize(letter,(32,32),interpolation = cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img,axis=0)
        character = classModel.predict(img)
        resultMaps.append(character)
    
    resultCharacters = []
    for i in resultMaps:
        resultCharacters.append(classes[np.argmax(i)])
        
    return [resultCharacters,resultSeg,resultMaps,segImg,perspective,canny,opening]
    
    