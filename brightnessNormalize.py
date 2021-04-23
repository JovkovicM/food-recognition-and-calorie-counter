import numpy as np
import cv2

ideal_brightness_for_increasing = 160.0
ideal_brightness_for_decreasing = 155.0

def decrease_brightnes(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    v[v < value] = 0
    v[v >= value] -= value.astype(np.uint8)
    
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def increase_brightness(img, value=60):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value.astype(np.uint8)
    
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def get_brightness(image):
    for_avg_light = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(for_avg_light)
    avg_light = v.mean()
    return avg_light

def brightness_normalize(image):
    brightness = get_brightness(image)
    #print("%d: light = %d"%(i,brightness))

    if(brightness<140):
        image = increase_brightness(image, ideal_brightness_for_increasing-brightness)
        #brightness = get_brightness(image)
        #print("%d: light[Fixed] = %d"%(i,brightness))
            
    if(brightness>170):
        image = decrease_brightnes(image, abs(ideal_brightness_for_decreasing-brightness))
        #brightness = get_brightness(image)
        #print("%d: light[Fixed] = %d"%(i,brightness))
    
    return image