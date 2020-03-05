
import numpy as np
import sys
import cv2
from copy import deepcopy
from PIL import Image
import pytesseract as tess
# sudo apt install tesseract-ocr
# pip install pytesseract

def preprocess(img, output_path):
        imgBlurred = cv2.GaussianBlur(img, (5,5), 0)
        cv2.imwrite(output_path + '/2_gaussian_blur.jpg', imgBlurred)

        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_path + '/3_gray.jpg', gray)

        sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
        cv2.imwrite(output_path + '/4_sobel.jpg', sobelx)

        ret2,threshold_img = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imwrite(output_path + '/5_threshold.jpg', threshold_img)
        return threshold_img

def cleanPlate(plate, output_path):
        print ("CLEANING PLATE. . .")
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        _,contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
                areas = [cv2.contourArea(c) for c in contours]
                max_index = np.argmax(areas)

                max_cnt = contours[max_index]
                max_cntArea = areas[max_index]
                x,y,w,h = cv2.boundingRect(max_cnt)

                if not ratioCheck(max_cntArea,w,h):
                    return plate,None 
                cleaned_final = thresh[y:y+h, x:x+w]
                return cleaned_final,[x,y,w,h]

        else:
                return plate,None


def extract_contours(threshold_img, output_path):
        element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
        morph_img_threshold = threshold_img.copy()
        cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
        cv2.imwrite(output_path + '/6_morphed.jpg', morph_img_threshold)

        _,contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
        return contours


def ratioCheck(area, width, height):
        ratio = float(width) / float(height)
        if ratio < 1:
                ratio = 1 / ratio

        aspect = 4.7272
        min = 15*aspect*15  # minimum area
        max = 125*aspect*125  # maximum area

        rmin = 3
        rmax = 6

        if (area < min or area > max) or (ratio < rmin or ratio > rmax):
                return False
        return True

def isMaxWhite(plate):
        avg = np.mean(plate)
        if(avg>=115):
                return True
        else:
                return False

def validateRotationAndRatio(rect):
        (x, y), (width, height), rect_angle = rect

        if(width>height):
                angle = -rect_angle
        else:
                angle = 90 + rect_angle

        if angle>15:
                return False

        if height == 0 or width == 0:
                return False

        area = height*width
        if not ratioCheck(area,width,height):
                return False
        else:
                return True



def cleanAndRead(img,contours, output_path):
        for i,cnt in enumerate(contours):
                min_rect = cv2.minAreaRect(cnt)

                if validateRotationAndRatio(min_rect):

                        x,y,w,h = cv2.boundingRect(cnt)
                        plate_img = img[y:y+h,x:x+w]


                        if(isMaxWhite(plate_img)):
                                clean_plate, rect = cleanPlate(plate_img, output_path)

                                if rect:
                                        x1,y1,w1,h1 = rect
                                        x,y,w,h = x+x1,y+y1,w1,h1
                                        cv2.imwrite(output_path + '/7_find_plate_using_contour_area.jpg', clean_plate)
                                        plate_im = Image.fromarray(clean_plate)
                                        text = tess.image_to_string(plate_im, lang='eng')
                                        print ("Detected Text : ",text)
                                        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                                        cv2.imwrite(output_path + '/8_detected_plate.jpg', img)



if __name__ == '__main__':
        if not sys.argv[1] and not sys.argv[2]:
            print ("Give arguments: python main.py <input image> <output images path>")
            sys.exit()
        print ("DETECTING PLATE . . .")

        input_img = sys.argv[1]
        output_path = sys.argv[2]

        img = cv2.imread(sys.argv[1])
        cv2.imwrite(output_path + '/1_input.jpg', img)

        threshold_img = preprocess(img, output_path)
        contours= extract_contours(threshold_img, output_path)

        cleanAndRead(img,contours, output_path)
