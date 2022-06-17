import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

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



def generate_results(dirname,predictions,masks,images_path):

    listdir = os.listdir(dirname)
    for dir in listdir:
        shutil.rmtree(dirname + dir)
    # reescale bin image
    os.mkdir(dirname + '!Todas')
    num = 0
    for index in tqdm(range(0,len(predictions))):
        pred = predictions[index].astype(np.uint8)
        mask = masks[index].numpy().astype(np.uint8)

        mask = cv2.resize(mask,(720,1160),interpolation = cv2.INTER_NEAREST)
        pred = cv2.resize(pred,(720,1160),interpolation = cv2.INTER_NEAREST)

        img = cv2.imread(images_path[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        res_pred = img
        pred[pred == 1] = 255
        # print(np.unique(pred))



        pred = pred.astype(np.uint8)

        contours,_ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # if len(contours)  == 0:
        #     break
        approx = cv2.approxPolyDP(contours[0], 0.04 * cv2.arcLength(contours[0], True), True)

        # rect = cv2.minAreaRect(contours[0])
        # box = cv2.boxPoints(rect)
        # print(box)
        # print(approx)
        box = np.squeeze(approx, axis=1)
        # cv2.drawContours(mask, [approx[:4]], 0, (255), 5)
        # for i in range(0,4):
        #     cv2.circle(mask,(box[i][0], box[i][1]), 25, (i+1)*50, -1)

        heigth = 70
        width = 180

        pt2 = np.float32([[0,0], [width,0], [0,heigth], [width,heigth]])
        # print(box)
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


        # for rho,theta in lines[0]:
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a*rho
        #     y0 = b*rho
        #     x1 = int(x0 + 1000*(-b))
        #     y1 = int(y0 + 1000*(a))
        #     x2 = int(x0 - 1000*(-b))
        #     y2 = int(y0 - 1000*(a))

        #     cv2.line(hough,(x1,y1),(x2,y2),(255),1)
        


        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(hough,(x1,y1),(x2,y2),(255),1)

        kernel = np.ones((int(heigth * 0.1)), dtype=np.uint8) * 255
        # plt.imshow(gray)

        opening = cv2.morphologyEx(hough, cv2.MORPH_OPEN, kernel)
        # plt.imshow(opening)
        boxop = getBoundingBoxFromMask(opening).astype(int)
        
        result = f2[boxop[0]:boxop[2],boxop[1]:boxop[3]]

        if not result.any():
            result = f2.copy()
        resultSeg = result.copy()

        he, wi,dm = resultSeg.shape
        div = int(wi/7)

        divs = [i * div + 5 for i in range(1,7)]
        divs[0] = divs[0] - 5
        divs[1] = divs[1] - 5
        divs[-2] = divs[-2] - 5
        divs[-1] = divs[-1] - 5

        # print(divs,width,resultSeg.shape)


        divImgList = [];

        for i in divs:
            # divImgList.append(resultSeg)
            resultSeg[:,i-1:i+1] = 0
            
        # plt.imshow(canny)
        # dir = dirname + str(index) + '_0' +  str(index2) + '/'
        # num = ((index + 1) * (index2 + 1))
        num = num + 1
        if num < 10:
            dir = dirname + '0' + str( num ) + '/'
        else:
            dir = dirname + str( num ) + '/'

        # dir = 'result/' + str(index) + '_0' +  str(index2) + '_'
        os.mkdir(dir)

        # ret, threshed_img = cv2.threshold(cv2.cvtColor(result,cv2.COLOR_BGR2GRAY),100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        # threshed_img = cv2.adaptiveThreshold(cv2.cvtColor(result,cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11 , 10)
        # ret, threshed_img = cv2.threshold(result[:,:,0],100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

        # cv2.imwrite(dir + '/'+ lista_dir[index][index2],cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        cv2.imwrite(dir + '1_original.jpg',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        cv2.imwrite(dir + '3_prediction.jpg',pred)
        cv2.imwrite(dir + '2_mask.jpg',mask * 255)
        cv2.imwrite(dir + '4_perspectiva.jpg',cv2.cvtColor(f2,cv2.COLOR_RGB2BGR))
        cv2.imwrite(dir + '5_canny.jpg',canny)
        cv2.imwrite(dir + '5.6_morphology.jpg',opening)
        if len(result) > 0:
            cv2.imwrite(dir + '6_recortadaCanny.jpg',cv2.cvtColor(result,cv2.COLOR_RGB2BGR))
            cv2.imwrite(dir + '7_recortadaCanny_CaracteresSegmentados.jpg',cv2.cvtColor(resultSeg,cv2.COLOR_RGB2BGR))
            cv2.imwrite(dirname + '!Todas/' + str(num) + '_7_recortadaCanny_CaracteresSegmentados.jpg',cv2.cvtColor(resultSeg,cv2.COLOR_RGB2BGR))
        # cv2.imwrite(dirname + '!Todas/' + str(index) + '_0' +  str(index2) + '_8_recortadaCanny_CaracteresSegmentados.jpg',cv2.cvtColor(threshed_img,cv2.COLOR_RGB2BGR))

def generate_results_reduced(dirname,predictions,masks,images_path):

    listdir = os.listdir(dirname)
    for dir in listdir:
        shutil.rmtree(dirname + dir)
    # reescale bin image
    os.mkdir(dirname + '00_all')
    num = 0
    for index in tqdm(range(0,len(predictions))):
        pred = predictions[index].astype(np.uint8)
        mask = masks[index].numpy().astype(np.uint8)

        mask = cv2.resize(mask,(720,1160),interpolation = cv2.INTER_NEAREST)
        pred = cv2.resize(pred,(720,1160),interpolation = cv2.INTER_NEAREST)

        img = cv2.imread(images_path[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        res_pred = img
        pred[pred == 1] = 255
        # print(np.unique(pred))



        pred = pred.astype(np.uint8)

        contours,_ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours)  > 0:
            approx = cv2.approxPolyDP(contours[0], 0.035 * cv2.arcLength(contours[0], True), True)

            # rect = cv2.minAreaRect(contours[0])
            # box = cv2.boxPoints(rect)
            # print(box)
            # print(approx)
            box = np.squeeze(approx, axis=1)
            # cv2.drawContours(mask, [approx[:4]], 0, (255), 5)
            # for i in range(0,4):
            #     cv2.circle(mask,(box[i][0], box[i][1]), 25, (i+1)*50, -1)

            heigth = 70
            width = 180

            pt2 = np.float32([[0,0], [width,0], [0,heigth], [width,heigth]])
            # print(box)
            if box[0][0] < box[2][0]:
                pt12 = np.float32([box[0],box[3],box[1],box[2]])
                if box[0][1] > box[2][1]:
                    pt12 = np.float32([box[3],box[2],box[0],box[1]])
            else:
                pt12 = np.float32([box[1],box[0],box[2],box[3]])
                if box[0][1] > box[2][1]:
                    pt12 = np.float32([box[2],box[1],box[3],box[0]])

            # if box[0][0] < box[2][0] & box[0][1] < box[2][1]:
            #     pt12 = np.float32([box[0],box[3],box[1],box[2]])
            # elif box[0][0] > box[2][0] & box[0][1] > box[2][1]: 
            #     pt12 = np.float32([box[1],box[0],box[2],box[3]])
            # else:
            #     pt12 = np.float32([box[2],box[1],box[3],box[0]])
                

            f = cv2.getPerspectiveTransform(pt12,pt2)
            f2 = cv2.warpPerspective(img,f,(width,heigth))


            resultSeg = f2.copy()

            he, wi,dm = resultSeg.shape
            div = int(wi/7)

            divs = [i * div + 5 for i in range(1,7)]
            divs[0] = divs[0] - 5
            divs[1] = divs[1] - 5
            divs[-2] = divs[-2] - 5
            divs[-1] = divs[-1] - 5
            # Used in the original segmentation masks
            # divs[0] = divs[0] - 5
            # divs[1] = divs[1] - 5
            # divs[-2] = divs[-2] - 7
            # divs[-1] = divs[-1] - 12

            # print(divs,width,resultSeg.shape)


            divImgList = [];

            for i in divs:
                # divImgList.append(resultSeg)
                resultSeg[:,i-1:i+1] = 0
                
            # plt.imshow(canny)
            # dir = dirname + str(index) + '_0' +  str(index2) + '/'
            # num = ((index + 1) * (index2 + 1))
            num = num + 1
            if num < 10:
                dir = dirname + '0' + str( num ) + '/'
            else:
                dir = dirname + str( num ) + '/'

            # dir = 'result/' + str(index) + '_0' +  str(index2) + '_'
            os.mkdir(dir)

            # ret, threshed_img = cv2.threshold(cv2.cvtColor(result,cv2.COLOR_BGR2GRAY),100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            # threshed_img = cv2.adaptiveThreshold(cv2.cvtColor(result,cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11 , 10)
            # ret, threshed_img = cv2.threshold(result[:,:,0],100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

            # cv2.imwrite(dir + '/'+ lista_dir[index][index2],cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            cv2.imwrite(dir + '1_original.jpg',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            cv2.imwrite(dir + '3_prediction.jpg',pred)
            cv2.imwrite(dir + '2_mask.jpg',mask * 255)
            cv2.imwrite(dir + '4_perspectiva.jpg',cv2.cvtColor(f2,cv2.COLOR_RGB2BGR))
            # if len(resultSeg) > 0:
            cv2.imwrite(dir + '7_recortada_CaracteresSegmentados.jpg',cv2.cvtColor(resultSeg,cv2.COLOR_RGB2BGR))
            cv2.imwrite(dirname + '00_all/' + str(num) + '_7_recortada_CaracteresSegmentados.jpg',cv2.cvtColor(resultSeg,cv2.COLOR_RGB2BGR))
            # cv2.imwrite(dirname + '!Todas/' + str(index) + '_0' +  str(index2) + '_8_recortadaCanny_CaracteresSegmentados.jpg',cv2.cvtColor(threshed_img,cv2.COLOR_RGB2BGR))



def generate_results_reduced(dirname,model,dataset):

    listdir = os.listdir(dirname)
    for dir in listdir:
        shutil.rmtree(dirname + dir)
    # reescale bin image
    os.mkdir(dirname + '00_all')
    num = 0
    for elem in tqdm(dataset):
        batch_size = len(elem[0])

        images_list = []
        for item in elem[0]: # images
            images_list.append(item)

        masks_list = []
        for item in elem[1]: # masks
            masks_list.append(item)

        index_img = 0
        pred_list = []
        perspective_list = []
        result_list = []
        predictions = model.predict(elem[0]) # predictions
        for item in predictions:
            pred = item > 0.5
            pred[pred == 1] = 255
            pred = pred.astype(np.uint8)
            contours,_ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            if len(contours)  > 0:
                approx = cv2.approxPolyDP(contours[0], 0.035 * cv2.arcLength(contours[0], True), True)

                # rect = cv2.minAreaRect(contours[0])
                # box = cv2.boxPoints(rect)
                # print(box)
                # print(approx)
                box = np.squeeze(approx, axis=1)
                # cv2.drawContours(mask, [approx[:4]], 0, (255), 5)
                # for i in range(0,4):
                #     cv2.circle(mask,(box[i][0], box[i][1]), 25, (i+1)*50, -1)

                heigth = 70
                width = 180

                pt2 = np.float32([[0,0], [width,0], [0,heigth], [width,heigth]])
                # print(box)
                if box[0][0] < box[2][0]:
                    pt12 = np.float32([box[0],box[3],box[1],box[2]])
                    if box[0][1] > box[2][1]:
                        pt12 = np.float32([box[3],box[2],box[0],box[1]])
                else:
                    pt12 = np.float32([box[1],box[0],box[2],box[3]])
                    if box[0][1] > box[2][1]:
                        pt12 = np.float32([box[2],box[1],box[3],box[0]])

                # if box[0][0] < box[2][0] & box[0][1] < box[2][1]:
                #     pt12 = np.float32([box[0],box[3],box[1],box[2]])
                # elif box[0][0] > box[2][0] & box[0][1] > box[2][1]: 
                #     pt12 = np.float32([box[1],box[0],box[2],box[3]])
                # else:
                #     pt12 = np.float32([box[2],box[1],box[3],box[0]])
                    
                f = cv2.getPerspectiveTransform(pt12,pt2)
                f2 = cv2.warpPerspective(images_list[index_img].numpy(),f,(width,heigth))
                index_img += 1

                resultSeg = f2.copy()

                he, wi,dm = resultSeg.shape
                div = int(wi/7)

                divs = [i * div + 5 for i in range(1,7)]
                divs[0] = divs[0] - 5
                divs[1] = divs[1] - 5
                divs[-2] = divs[-2] - 5
                divs[-1] = divs[-1] - 5
                # Used in the original segmentation masks
                # divs[0] = divs[0] - 5
                # divs[1] = divs[1] - 5
                # divs[-2] = divs[-2] - 7
                # divs[-1] = divs[-1] - 12

                # print(divs,width,resultSeg.shape)


                divImgList = [];

                for i in divs:
                    # divImgList.append(resultSeg)
                    resultSeg[:,i-1:i+1] = 0
                
                pred_list.append(pred) 
                perspective_list.append(f2)
                result_list.append(resultSeg)
        
        for index in range(0,batch_size):
            num = num + 1
            if num < 10:
                dir = dirname + '0' + str( num ) + '/'
            else:
                dir = dirname + str( num ) + '/'
            
    
            os.mkdir(dir)

            cv2.imwrite(dir + '1_original.jpg',images_list[index].numpy() * 255)
            cv2.imwrite(dir + '3_prediction.jpg',pred_list[index] * 255)
            cv2.imwrite(dir + '2_mask.jpg',masks_list[index].numpy() * 255)
            cv2.imwrite(dir + '4_perspectiva.jpg',perspective_list[index] * 255)
            cv2.imwrite(dir + '7_recortada_CaracteresSegmentados.jpg',result_list[index] * 255)
            cv2.imwrite(dirname + '00_all/' + str(num) + '_7_recortada_CaracteresSegmentados.jpg',result_list[index] * 255)
