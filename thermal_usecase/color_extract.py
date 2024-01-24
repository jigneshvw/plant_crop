import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#!pip install easydev                 #version 0.12.0
#!pip install colormap                #version 1.0.4
#!pip install opencv-python           #version 4.5.5.64
#!pip install colorgram.py            #version 1.2.0
#!pip install extcolors               #version 1.0.0

import cv2
import extcolors

from colormap import rgb2hex
import pickle



# Load pickle dictionary data (deserialize)
with open('category_classify_color.pickle', 'rb') as handle:
    category_classify_color = pickle.load(handle)



def get_color_name(rgb):
	colors = {
	    "red": (255, 0, 0),
	    # "pink": (255, 192, 203), ----(no need to include this color)
	    "maroon": (128, 0, 0),
	    "green": (0, 255, 0),
	    "blue": (0, 0, 255),
	    "yellow": (255, 255, 0),
	    # "yellow (gold)": (255,215,0), #if used this, then comment above yellow
	    "magenta": (255, 0, 255),
	    "cyan": (0, 255, 255),
	    "black": (0, 0, 0),
	    "white": (255, 255, 255),
	    "orange": (255, 165, 0),
	    # "orange (dark orange)": (255,140,0), #if used this, then comment above orange
	    "purple": (160, 32, 240),
	    "violet": (143, 0, 255),
	    "indigo": (75, 0, 130)
	}
	min_distance = float("inf") #inf is for infinity value placeholder
	closest_color = None
	for color, value in colors.items():
	    distance = sum([(i - j) ** 2 for i, j in zip(rgb, value)])
	    if distance < min_distance:
	        min_distance = distance
	        closest_color = color
	return closest_color



def color_to_df(input):
    colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
    print(colors_pre_list)
    df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
    print('df_rgb',df_rgb)
    df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]
    
    # convert RGB to HEX code (use below code to convert rgb into hex code and use for labelling)
    df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
                          int(i.split(", ")[1]),
                          int(i.split(", ")[2].replace(")",""))) for i in df_rgb]


	# below code is to only get access to RGB code values and store inside list
    df_rgb_val = [(int(i.split(", ")[0].replace("(","")),
                          int(i.split(", ")[1]),
                          int(i.split(", ")[2].replace(")",""))) for i in df_rgb]


    # get_color_name() is used to convert rgb code value into actual color name
    df_rgb_color = [get_color_name(col) for col in df_rgb_val]

    df_rgb_color_category = [category_classify_color[color] for color in df_rgb_color]

    df = pd.DataFrame(zip(df_color_up, df_percent, df_rgb_val, df_rgb_color, df_rgb_color_category), columns = ['c_code','occurence', 'rgb_val', 'rgb_color', 'rgb_color_category'])
    return df


def extract_color_category(input_image, resize, tolerance, zoom):
    #background
    bg = 'bg.png'
    fig, ax = plt.subplots(figsize=(192,108),dpi=10)
    fig.set_facecolor('white')
    plt.savefig(bg)
    plt.close(fig)
    
    #resize
    output_width = resize
    img = Image.open(input_image)
    if img.size[0] >= resize:
        wpercent = (output_width/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((output_width,hsize), Image.ANTIALIAS)
        resize_name = 'resize_'+ input_image
        img.save(resize_name)
    else:
        resize_name = input_image
    
    #create dataframe
    img_url = resize_name
    colors_x = extcolors.extract_from_path(img_url, tolerance = tolerance, limit = 13)
    print(colors_x)
    df_color = color_to_df(colors_x)
    df_color['c_code'] = df_color['c_code']

    print(df_color)

    
    #annotate text
    list_colors_name = list(df_color['rgb_color_category'])

    list_precent = [int(i) for i in list(df_color['occurence'])]
    text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_colors_name, list_precent)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160,120), dpi = 10)
    
    #donut plot
    list_color = list(df_color['c_code'])
    print('text_c',text_c)
    print('list_color',list_color)
    wedges, text = ax1.pie(list_precent,
                           labels= text_c,
                           labeldistance= 1.05,
                           colors = list_color,
                           textprops={'fontsize': 150, 'color':'black'})
    plt.setp(wedges, width=0.3)

    #add image in the center of donut plot
    img = mpimg.imread(resize_name)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (0, 0))
    ax1.add_artist(ab)
    
    #color palette
    x_posi, y_posi, y_posi2 = 160, -170, -170
    for c in list_color:
        if list_color.index(c) <= 5:
            y_posi += 180
            rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor = c)
            ax2.add_patch(rect)
            ax2.text(x = x_posi+400, y = y_posi+100, s = c, fontdict={'fontsize': 190})
        else:
            y_posi2 += 180
            rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor = c)
            ax2.add_artist(rect)
            ax2.text(x = x_posi+1400, y = y_posi2+100, s = c, fontdict={'fontsize': 190})

    fig.set_facecolor('white')
    ax2.axis('off')
    bg = plt.imread('bg.png')
    plt.imshow(bg)       
    plt.tight_layout()
    return plt.show()



if __name__=="__main__":

# extract_color_category('image name', resized_width, tolerance, zoom) #parameter/argument definition

	extract_color_category(r"D:\plant_proj\thermal_usecase\Thermal Image\101000101.jpg", 900, 12, 2.5)
	# print(get_color_name((235, 187, 15))) this is just to run separately and check the output of get_color_name()

