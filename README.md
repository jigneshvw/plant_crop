# plant_crop
Three stages of plant crop project: 
1) predict disease using normal image, 
2) predict growth stage, 
3) predict disease using thermal image and color pattern identification

# More details:
1) Plant disease and deficiency prediction is done using the RGB (visual) leaf image data of plant/crop. Based on trained model using custom model with pytorch and CNN architecture (keras and tensorflow), multi layered model is designed that is trained on image dataset. Once deployed, the model should be able to detect and identify type of disease based on plant leaf with the distance metric (accuracy). 
2) Growth stage prediction involves monitoring and calculating the growth of plant/crop during different intervals of time. Based on the image of plant, the height and width is measured of plant. Plantcv, Opencv and other python libraries are used for this, along with some rule based method based on image atributes and features to calculate height and width.
3) Predict disease using thermal image has same method as first point, but this is done on thermal infrared images, unlike rgb image. Also, range of different color patterns in thermal image is identified, along with the weightage of each color embedded in an image.

===Major languages/tools used:
~ Python (programming), 
~ SQL (DML operations),
~ Microsoft SQL server (for database),
~ Linux based C-panel for hosting,
~ Flask (api build and connection), etc

