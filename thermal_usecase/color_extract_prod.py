import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os

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

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flask import Flask, render_template, Response
# from flask import redirect, url_for

import pymysql 


# * ---------- Create App --------- *
app = Flask(__name__)
CORS(app, support_credentials=True)

# @app.route('/')
# def index():
#   return render_template('index.html')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


FILE_PATH = os.path.dirname(os.path.realpath(__file__))
print(FILE_PATH)

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




@app.route('/thermal_pred',methods=['GET','POST'])
# def extract_color_category(tolerance, zoom):
def extract_color_category():

   # To connect MySQL database 
    conn = pymysql.connect( 
        host='localhost', 
        user='username', 
        db='DB name', 
        password='password',
        port='port number'
        ) 
      
    cur = conn.cursor() 
    cur.execute("select * from thermal_plant where processed_new = 'new'") 
    output_new_data = cur.fetchall() 
    # folder_name = output[0][1]
    # train_data_path = output[0][2]

    print('len output_new_data::: ', output_new_data)

    cur.execute("select * from thermal_plant") 
    output_all_data = cur.fetchall() 

    print('len output_all_data::: ', output_all_data)

    # To close the connection 
    conn.close() 

    source_path = output_new_data[0][2]


    try:
        #create dataframe
        all_subfolders = []
        # image_path = '{path}/Data/Crop1-tomato/test/{image_folder}/{test}'.format(path=FILE_PATH, test=input_id, image_folder=folder_name_image) #testimages[i]----- (query image) change path as per server      
        # image_path = '{path}\\thermal_test\\'.format(path=FILE_PATH) #testimages[i]----- (query image) change path as per server
        image_path = '{path}'.format(path=source_path) #testimages[i]----- (query image) change path as per server

        for i in os.listdir(image_path):
            path = image_path + i
            all_subfolders.append(path)

        sub_fol_image_path_dict = {}
        for subdir in all_subfolders:
            all_image_path = []
            for (dirpath, dirnames, filenames) in os.walk(subdir): #change path as per the server
                for img_name in filenames:
                    image_names = dirpath + '\\' + img_name
                    all_image_path.append(image_names)
            sub_fol_image_path_dict[subdir.split('\\')[-1]] = all_image_path

        all_preds_text_c = {}  
        all_preds_list_color = {}  

        for ind, dir_name in enumerate(sub_fol_image_path_dict.keys()):
            print('dir_name', dir_name) 

            pred_values_text_c = {}

            pred_values_list_color = {}

            for ind, single_path in enumerate(sub_fol_image_path_dict[dir_name]):

                colors_x = extcolors.extract_from_path(single_path, tolerance = 12, limit = 13)
                df_color = color_to_df(colors_x)
                df_color['c_code'] = df_color['c_code']
                
                #annotate text
                list_colors_name = list(df_color['rgb_color_category'])

                list_precent = [int(i) for i in list(df_color['occurence'])]
                text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_colors_name, list_precent)]
                
                #donut plot
                list_color = list(df_color['c_code'])
                print('text_c',text_c)
                print('list_color',list_color)
                
                print('=====', single_path.split('\\')[-1])
                pred_values_text_c[single_path.split('\\')[-1]] = text_c #store all image text_c pred name and value inside each subfolder
                pred_values_list_color[single_path.split('\\')[-1]] = list_color #store all image list_color pred name and value inside each subfolder
                
            all_preds_text_c[dir_name] = pred_values_text_c #store pred_values_sub_fol subfolders name wise
            all_preds_list_color[dir_name] = pred_values_list_color #store pred_values_sub_fol subfolders name wise
        
        print(all_preds_text_c)
        print(all_preds_list_color)

        len_output_all_data = len(output_all_data)
        len_output_new_data = len(output_new_data)
        # generate few featured attributes to insert the value in already saved values in DB of predicted outcome of plant data
        for ind, ech_plant in enumerate(output_new_data):
            len_output_all_data = len_output_all_data
            len_output_new_data = len_output_new_data
            if ind == 0:
                len_output_all_data = (len_output_all_data - len_output_new_data) + 1
            else:
                len_output_all_data = len_output_all_data + 1

            conn = pymysql.connect( 
                host='localhost', 
                user='root', 
                db='plant', 
                password='1234',
                port=3307
                ) 

            conn.begin()
            cur = conn.cursor() 
            print(ech_plant)
            exec_statement = ' UPDATE thermal_plant SET id= {class_name} , results_colorcat_weight= "{output_res_colorcat}", results_hex= "{output_res_hex}", processed_new = "{predict_old_new}", status = "{status_db}"  WHERE folder_name = "{fold_name}"'.format(class_name = len_output_all_data, output_res_colorcat = all_preds_text_c[list(all_preds_text_c.keys())[ind]], output_res_hex = all_preds_list_color[list(all_preds_list_color.keys())[ind]], fold_name = ech_plant[1], predict_old_new = 'processed', status_db = 'success')
            print(exec_statement)
            cur.execute(exec_statement)
            conn.commit() 

        # conn.rollback() #if you want to go back to previous values in table
        conn.close()    
        print('database updated!!---')

        answer = 'success' #success means atleast one result found while predicting
        print(answer)

    except:

        answer = 'error'
        print(answer)

    return jsonify(answer)


if __name__=="__main__":

    app.run(host='127.0.0.1', port=5000, debug=True)

# extract_color_category('image name', resized_width, tolerance, zoom) #parameter/argument definition

    # extract_color_category(12, 2.5)
    extract_color_category()
	# print(get_color_name((235, 187, 15))) this is just to run separately and check the output of get_color_name()

