import numpy as np
import cv2

table_orange_lower = (10, 40, 170)
table_orange_upper = (80, 170, 256)

shadow_black_lower = (0, 0, 180)
shadow_black_upper = (180, 30, 230)

lower = {'white': (0, 0, 140), 'pink': (140, 110, 100)}
upper = {'white': (180,70,256), 'pink': (180, 200, 256)}
 
lower_plate = {'white': (0, 0, 80), 'pink': (140, 110, 100)}
upper_plate = {'white': (180,100,256), 'pink': (180, 200, 256)}

def create_circular_mask(h, w, center=None, radius=None): 
    #print(h,w,center,radius)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask


def getCircle(key, image):
    kernel = np.ones((9,9),np.uint8)
    mask1 = cv2.inRange(image, lower[key], upper[key])
    mask2 = cv2.inRange(image, table_orange_lower, table_orange_upper)
    mask2_inv = cv2.bitwise_not(mask2)
    multi_mask = cv2.bitwise_and(mask1, mask2_inv)

    mask = cv2.morphologyEx(multi_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    #for c in cnts:
    #   peri = cv2.arcLength(c, True)
    #   approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    #   cv2.drawContours(image, [approx], -1, (60, 255, 100), 2)
    
    #center = None
    
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        #M = cv2.moments(c)
        #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        #cv2.circle(image, (int(x), int(y)), int(radius), colors[key], 2)
        #image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        #cv2.imshow('imageasd%d' %(i), image)

        if radius > 140:
            return (int(x), int(y), int(radius))
        
    return (int(0), int(0), int(0))

def separate_plate_from_background(image):
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
     
    (x,y,radius) = getCircle("pink", hsv)

    if(radius == 0):
        (x,y,radius) = getCircle("white", hsv)
    
    if(radius == 0):
        print("brisem je") #TODO

    height = image.shape[0]
    width = image.shape[1]
    mask = create_circular_mask(height, width, (x,y), radius)
    only_plate = np.zeros_like(image) # Extract out the object and place into output image
    only_plate[mask == True] = image[mask == True]
    
    return only_plate

def get_food(plate_image):
    """
    dilated_img = cv2.dilate(plate_image, np.ones((7,7), np.uint8))
    cv2.imshow('dilated_img%d' %(1), dilated_img)
    bg_img = cv2.medianBlur(dilated_img, 21)
    cv2.imshow('bg_img%d' %(1), bg_img)
    diff_img = 255 - cv2.absdiff(plate_image, bg_img)
    cv2.imshow('diff_img%d' %(1), diff_img)
    norm_img = diff_img.copy()
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imshow('norm_img%d' %(1), norm_img)
    """
    
    only_plate_HSV = cv2.cvtColor(plate_image,cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(only_plate_HSV, lower_plate["white"], upper_plate["white"])
    mask2 = cv2.inRange(only_plate_HSV, lower_plate["pink"], upper_plate["pink"])
    mask3 = cv2.inRange(only_plate_HSV, shadow_black_lower, shadow_black_upper)

    only_food = np.zeros_like(only_plate_HSV)
    only_food[((mask1 == 0) & (mask2 == 0))] = only_plate_HSV[((mask1 == 0) & (mask2 == 0))]
    
    mask1_inv = cv2.bitwise_not(mask1)
    masked_data = cv2.bitwise_and(plate_image, plate_image, mask=mask1_inv)
    mask2_inv = cv2.bitwise_not(mask2)
    masked_data = cv2.bitwise_and(masked_data, masked_data, mask=mask2_inv)
    
    mask3_inv = cv2.bitwise_not(mask3)
    masked_data = cv2.bitwise_and(masked_data, masked_data, mask=mask3_inv)
    #cv2.imshow('masked_data%d' %(1), masked_data)
    
    """
    edged = cv2.Canny(masked_data, 70, 90, 9)
    #cv2.imshow('edged%d' %(i), edged)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('closed%d' %(i), closed)

    #finding_contours 
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print (cnts)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        cv2.drawContours(plate_image, [approx], 0, (60, 255, 100), 2)
    #cv2.imshow('drawContours%d' %(i), plate_image)
    """
        
    fruit_bw = cv2.cvtColor(masked_data, cv2.COLOR_BGR2GRAY)
    fruit_bin = cv2.inRange(fruit_bw, 10, 255) #binary of fruit

    kernel_git = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erode_fruit = cv2.erode(fruit_bin,kernel_git,iterations = 1)
    #cv2.imshow('erode_fruit', erode_fruit)
    #cv2.waitKey(0)

    #find largest contour since that will be the fruit
    img_th = cv2.adaptiveThreshold(erode_fruit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask_fruit = np.zeros(fruit_bin.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    cv2.drawContours(mask_fruit, [largest_areas[-2]], 0, (255,255,255), -1)
    #cv2.imshow('mask_fruit%d' %(1), mask_fruit)
    
    #find area of fruit
    fruit_contour = largest_areas[-2]
    fruit_area = cv2.contourArea(fruit_contour)
    #print(fruit_area)
    
    #dilate now
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
    mask_fruit2 = cv2.dilate(mask_fruit,kernel2,iterations = 1)
    #res = cv2.bitwise_and(fruit_bin,fruit_bin,mask = mask_fruit2)
    #cv2.imshow('res%d' %(1), res)
    fruit_final = cv2.bitwise_and(plate_image,plate_image,mask = mask_fruit2)
    #cv2.imshow('fruit_final%d' %(1), fruit_final)

    
    return fruit_final, fruit_area, fruit_contour