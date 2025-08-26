# Uploading-a-photo-and-captions-to-it
In this work, we use kagglehub to download the Flickr8k dataset, which contains images and their descriptions. Image paths and text descriptions are loaded from the captions.txt file using pandas. A pre-trained ResNet50 model is used to extract image features. The extract_features function extracts features from images by resizing them to 224x224 and applying preprocessing. To save resources, we set a limit and extract features for the first 1,000 images. The text descriptions are tokenized and converted into sequences of numbers. Matrices are created for the input and target data of the decoder. A model is created that accepts image features and text sequences and is trained to predict the next word in the description. The model is trained on the data using RMSprop and cross-entropy. The last step is to visualize the result.

***For correct operation, you must install:***

%%capture

! pip install keras

! pip install tensorflow

! pip install datasets==3.6.0
