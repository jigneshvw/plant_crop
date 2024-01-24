from detecto import core, utils, visualize
import detecto
from torchvision import transforms
import matplotlib.pyplot as plt


data_xml=detecto.utils.xml_to_csv(r'D:\plant_proj\growth_prediction\new_data\xml_data_train\revised_data', 'train.csv')
data_xml.reset_index(drop=True, inplace=True)
data_xml

data_val=detecto.utils.xml_to_csv(r'D:\plant_proj\growth_prediction\new_data\xml_data_valid\xml_images', 'train.csv')
data_val.reset_index(drop=True, inplace=True)


data_val.to_csv(r'D:\plant_proj\growth_prediction\new_data\xml_data_train\train.csv', index=False) #save to use later while model training


data_val.to_csv(r'D:\plant_proj\growth_prediction\new_data\xml_data_valid\valid.csv', index=False) #save to use later while model training


data_xml.info()


#below are codes to copy image from one folder to another:

# f_names=data_xml[data_xml['class'].apply(lambda x:x.startswith('large'))==False]['filename']

# all_f_names = []
# for name in f_names:
#     all_f_names.append(f'D:\plant_proj\growth_prediction\data\train\large\{name}')
# all_f_names


# import glob
# files = glob.glob('D:/plant_proj/growth_prediction/data_xml/train_xml/*.xml', 
#                    recursive = True)
# len(files)

# import shutil, os
# files = all_f_names
# for f in files:
#     shutil.copy(f, 'D:/plant_proj/growth_prediction/data_xml/train_xml')



# #mention your dataset path
# data_train = core.Dataset(r'D:\plant_proj\growth_prediction\new_data\xml_data_train\revised_data')
# data_valid = core.Dataset(r'D:\plant_proj\growth_prediction\new_data\xml_data_valid\xml_images')

# label=list(range(1,data_xml.shape[0]+1)) #generate labels

# #mention your object label here
# model = core.Model(classes=label, model_name='fasterrcnn_mobilenet_v3_large_fpn')
# model.fit(dataset=data_train, epochs = 10)



# Advanced


# Convert XML files to CSV format
utils.xml_to_csv(r'D:\plant_proj\growth_prediction\new_data\xml_data_train\revised_data', 'train_labels.csv')
utils.xml_to_csv(r'D:\plant_proj\growth_prediction\new_data\xml_data_valid\xml_images', 'val_labels.csv')

# Define custom transforms to apply to your dataset
custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(800),
    transforms.ColorJitter(saturation=0.3),
    transforms.ToTensor(),
    utils.normalize_transform(),
])


# Pass in a CSV file instead of XML files for faster Dataset initialization speeds
dataset = core.Dataset(r'D:\plant_proj\growth_prediction\new_data\xml_data_train\train.csv', r'D:\plant_proj\growth_prediction\new_data\xml_data_train\images', transform=custom_transforms)
val_dataset = core.Dataset(r'D:\plant_proj\growth_prediction\new_data\xml_data_valid\valid.csv', r'D:\plant_proj\growth_prediction\new_data\xml_data_valid\images')  # Validation dataset for training

# dataset = core.Dataset(r'D:\plant_proj\growth_prediction\new_data\xml_data_train\revised_data', transform=custom_transforms)
# val_dataset = core.Dataset(r'D:\plant_proj\growth_prediction\new_data\xml_data_valid\xml_images')  # Validation dataset for training

# Create your own DataLoader with custom options
loader = core.DataLoader(dataset, batch_size=10, shuffle=True)

label=list(range(1,data_xml.shape[0]+1)) #defining label classes

# # To generate and dump labels used while training, this will be loaded while testing model
# import pickle
# with open(r'D:\plant_proj\growth_prediction\labels.pickle', 'wb') as classes:
#     pickle.dump(label, classes)

# Use MobileNet instead of the default ResNet
model = core.Model(label, model_name='fasterrcnn_mobilenet_v3_large_fpn' ) #or use  model_name='fasterrcnn_mobilenet_v3_large_fpn' as parameter
losses = model.fit(loader, val_dataset, epochs=50, learning_rate=0.0001, verbose=True, lr_step_size=5) 
#losses = model.fit(loader, val_dataset, epochs=20, learning_rate=0.001, 
#                    lr_step_size=5, verbose=True) # for epochs=20, learning_rate=0.001


plt.plot(losses)  # Visualize loss throughout training
plt.show()



image = utils.read_image(r"D:\plant_proj\growth_prediction\new_data\xml_data_valid\images\2.jpg")  # Helper function to read in images

labels, boxes, scores = model.predict(image)  # Get all predictions on an image
# predictions = model.predict_top(image)  #output gives: labels, boxes, scores
predictions = model.predict(image)#output gives: labels, boxes, scores


print(labels, boxes, scores)

print('predictions: ', predictions) #output gives: labels, boxes, scores

# visualize.show_labeled_image(image, boxes, labels)
visualize.show_labeled_image(image, predictions[1]) 

# model.save('D:/plant_proj/growth_prediction\model_weights.pth')  # Save model to a file

# Directly access underlying torchvision model for even more control
# torch_model = model.get_internal_model()
# print(type(torch_model))


print('model.get_internal_model: ', model.get_internal_model)

