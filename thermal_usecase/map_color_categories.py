#this code is to map each color with its respeective class/categories and save this dictionary file in local as pickle file

import pickle


category_classify_color = {"red": 'red (PHE-UID02A)',
# "pink": 'pink (PHE-UID02A)', ----(no need to include this color)
"maroon": 'maroon (PHE-UID02A)',
"green": 'green',
"blue":  'blue (PHE-UID01B)',
"yellow": 'yellow (PHE-UID01A)',
# "yellow (gold)": 'yellow (PHE-UID01A)', #if used this, then comment above yellow
"magenta": 'magenta (PHE-UID02A)',
"cyan": 'cyan (PHE-UID01B)',
"black": 'black',
"white": 'white',
"orange": 'orange (PHE-UID03)',
# "orange (dark orange)": 'orange (PHE-UID03)', #if used this, then comment above orange
"purple": 'purple (PHE-UID02B)',
"violet": 'violet (PHE-UID02B)',
"indigo": 'indigo (PHE-UID01B)'}



with open(r'D:\plant_proj\thermal_usecase\category_classify_color.pickle', 'wb') as handle:
    pickle.dump(category_classify_color, handle)
