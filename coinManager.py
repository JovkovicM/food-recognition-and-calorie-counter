import cv2
import numpy as np


max_coin_dim = 90
max_dims_dif = 15
min_coin_dim = 40

default_coin_radius = 32

lower_coin = (0, 50, 50)
upper_coin = (180, 150, 150)


def getCoinSize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(9,9),0)
    edge = cv2.Canny(blurred,20,90) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    #edge = cv2.dilate(edge,kernel,iterations = 1)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
    
    (cnts,_) = cv2.findContours(edge.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    coins = image.copy()
   
    coin_contures = []
    for i in range(0, len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[i])
        cv2.fillPoly(edge, pts =[cnts[i]], color=(0,255,0))

        peri = cv2.arcLength(cnts[i], True)
        approx = cv2.approxPolyDP(cnts[i], 0.01 * peri, True)

        cv2.fillPoly(edge, pts =[approx], color=(0,0,255))
    
        if(w +max_dims_dif>h):
            if(w-max_dims_dif<h):
                if(h+max_dims_dif>w):
                    if(h-max_dims_dif<w):
                        if(h>min_coin_dim):
                            if(w>min_coin_dim):
                                if(w<max_coin_dim):
                                    if(h<max_coin_dim):
                                        coin_contures.append(approx)
                                        
                                        

            
    if len(coin_contures) > 0:
        c = max(coin_contures, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        return round((h+w)/4)
    else:
        return default_coin_radius
     
def testCoinSizeAccuracy(dataset_coin_radius_pixels, my_coin_radius_pixels):       
    num_of_hits = 0
    bigger = 0
    lower = 0
    for i in range(len(dataset_coin_radius_pixels)):
        print(dataset_coin_radius_pixels[i], my_coin_radius_pixels[i])

        if(dataset_coin_radius_pixels[i]+5>=my_coin_radius_pixels[i]):
            if(dataset_coin_radius_pixels[i]-5<=my_coin_radius_pixels[i]):
                num_of_hits +=1
            else:
                print("lower")
                lower +=1
        else:
            print("bigger")
            bigger +=1
        

    print(np.mean(dataset_coin_radius_pixels))
    print(np.mean(my_coin_radius_pixels))

    print("hited = ", (num_of_hits/len(dataset_coin_radius_pixels))*100, "%")
    print("lower = ", (bigger/len(dataset_coin_radius_pixels))*100, "%")
    print("bigger = ", (lower/len(dataset_coin_radius_pixels))*100, "%")

