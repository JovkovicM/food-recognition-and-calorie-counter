import numpy as np
import cv2

coin_multiplier = 2.5

food_type_to_feature = dict()

def read_food_features():
    import csv 
    with open('Food_info.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        next(reader, None)
        for row in reader:
            food_type = row[1]
            food_type_to_feature[food_type] = list()
            food_type_to_feature[food_type].append(row[2])
            food_type_to_feature[food_type].append(float(row[3]))
            food_type_to_feature[food_type].append(float(row[4]))
            food_type_to_feature[food_type].append(float(row[5]))
    #print(food_type_to_feature)


def get_calorie(type_of_food, volume): #volume in cm^3
    calorie = food_type_to_feature[type_of_food][2]
    density = food_type_to_feature[type_of_food][1]
    
    mass = volume*density
    calorie_tot = (calorie/100.0)*mass
    
    return mass, calorie_tot #calorie per 100 grams

def get_volume(type_of_food, food_area_px, fruit_contour, coin_area_px, pix_to_cm_multiplier):
    food_area_real = (food_area_px/coin_area_px)*coin_multiplier #area in cm^2
    
    shape = food_type_to_feature[type_of_food][0]
    volume = 0
    
    if shape == "sphere": #sphere-apple,tomato,orange,kiwi,onion
        radius = np.sqrt(food_area_real/np.pi)
        volume = (4/3)*np.pi*radius*radius*radius

    if shape == "ellipsoid": #ellipsoid like banana, cucumber, carrot
        fruit_rect = cv2.minAreaRect(fruit_contour)
        a = (max(fruit_rect[1])*pix_to_cm_multiplier)/2 #PROVERITI: da li je ovo najudaljenija (a) ili najbliza duz (b)
        b = food_area_real/(a*np.pi)
        #print("a = ", a, ", b = ", b)
        c = b
        volume = (4/3)*np.pi*a*b*c

    if shape == "cuboid": #cuboid like bread, sachima
        height = np.sqrt(food_area_real)*food_type_to_feature[type_of_food][3]
        volume = food_area_real*height
        
    if shape == "cylinder": #cylinder like mooncake
        radius = np.sqrt(food_area_real/np.pi)
        height = radius*food_type_to_feature[type_of_food][3]
        print("r = ", radius, ", h = ", height)
        volume = np.pi*radius*radius*height

    return volume
    
def get_coin_info(coin_radius = 32):
    coin_area = coin_radius*coin_radius*np.pi
    pix_to_cm_multiplier = 2.5/(coin_radius*2)
    
    return coin_area, pix_to_cm_multiplier