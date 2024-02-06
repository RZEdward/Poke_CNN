# Plan:
# Download and import dataset of 150 Gen-1 Pokemon - 25-50 Samples of Each Pokemon
# Organise as ( Sample 1, Sample 2, Sample 3, ... ) , ( Label 1, Label 2, Label 3, ...) - then one-hot-encode the labels
# Each sample actually becomes pixel data with w*h pixels x 3 (for 3 layers of RGB) - so say 1 sample = 128x128x3 matrix

# Split samples/labels into Training, Test, Validation data - Should this be random across the dataset or random and proportional for each label (pokemon)?

# Define our model layers using Sequential() from Keras: add convolution layers, pooling layers, activation etc.
# Run Convolutional Neural Network
# Measure accuracy, F1, recall, precision etc (for total dataset and each pokemon, and each type/colour?)
# Print a single example from the testing data with true label and predicted label
# Publish results (from overleaf) - explain methodology, results, background math

# CNN Notes:

# CNN - artificial neural network with specialization in hidden layer to find image patterns
# Separated from MLP (multi layer perceptron) by hidden convolutional layers (uses convolution operation) - often also have non-convolutional layers
# In early layers - convolution matrices may convolve through the pixel matrix to identify left, right, top, bottom edges of shapes - bright/positive results in the ouput matrix indicate presence of targeted feature
# Later, more complex layers use a kernel (matrix operator cycling through input matrix of pixel data) to identify more complex patterns e.g circles, corners - there may be many convolution layers in total, 10s to 100s to 1000s
# Layers may recognise colours or directions (ie if there are regular streaks/sketch marks/lines across the horizontal) 
# at the highest levels of complexity layers can recognise combinations of coloured edges/curves/directions/patterns

# When going from grayscale to colour images - you need to turn your wxh matrix into wxh x 3 (for RGB) so you have 3 input matrices of RGB intensity
# you end up convolving with a 3D matrix/cube - note that the cube shifts down a row once RHS meets edge of input matrix (depending on padding, stride variables), so for input matrix of NxN you output a matrix of (N-2)x(N-2) (again, if you didnt use padding)

# The ultimate aim is to backtest until you find the convolution kernel(s) that most frequently/accurately convert a sample matrix into the desired matrix (that represents a the attached label / given pokemon)

# purely mathematically, output vectors/matrices are longer/bigger than than the sum of 2 convolved input matrices as they consider partial overlaps between the convolutional matrix and input matrix
# sometimes it is best to only consider full overlaps ( so a 6x6 goes to a 4x4 for example )
# in pure math, you 'flip around'/180 rotation the convolution matrix

# note scipy fft convolve is computationally faster than np.convolve - fft follows from (a, bx, cx2)(d, ex, fx2) expanding out concept / summing of diagonals - in reality you treat the each of the values of 2 input vectors you wish to convolve as polynomial coefficients, then take a finite sample of points on the 2 polynomials, multiply them point wise, then solve a linear system of simultaneous equations in the sampled convolved polynomial output to recover the full convolution
# by choosing to take samples at a specific set of complex numbers - you generate redundancy of coefficients and engineer the system to mirror FFT input - enabling FFT to govern this computation

# As well as convolution layers there are pooling layers - pool matrix is also 3x3 and selects max value within a window

# Normalise RGB values (/255) - this is key
# Ensure greyscale is defined as 0 = black, 1 = white

# Use flattened entry so each row is 1 image with 1 label, rather than having 1 row as a group of labels
# Later we will need 1 row = wxh pixels as the features - so 1 row per image

# might be interesting to see accuracy per pokemon vs n_samples for that pokemon - 'To what degree does more samples increase model accuracy'

# ---------------------------------------------------------------------------------------------------------

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

dataset_path = 'PokemonData' # Folder of all folders of images
image_data = []
labels = []

resize_dim = (256,256) # Here is the image size we will apply to all images - need to be uniform

Count = 0
Pokemon = 0
for folder_name in os.listdir(dataset_path):
    Pokemon += 1
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            Count +=1
            file_path = os.path.join(folder_path, file_name)
            try:
                if file_name.endswith('.svg'): # can't recognise SVGs, could convert them but there are only 13/6238
                    continue
                with Image.open(file_path) as img:
                    img = img.convert('RGB')  # break down the image into a matrix of RGB pixel values
                    img_resized = img.resize(resize_dim, Image.Resampling.BILINEAR)
                    img_array = np.array(img_resized) / 255.0  # Normalization step
                    image_data.append(img_array)
                    labels.append(folder_name) # The names of folders are Pokemon names and so become the labels - later we will convert names to number classification
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")

image_data = np.array(image_data)
labels = np.array(labels)
#print(image_data.shape)
#print(len(labels)) , expect this to be equal to the total number of images

print('Total Unique Pokemon in Storage: ', Pokemon)
print('Total Unique Labels Loaded Into Dataset: ', len(set(labels))) # if this differs from above line then some pokemon didnt have any images (or images didnt load for given pokemon)
print('Total Images in Storage: ', Count)
print(f"Total Images Loaded Into Dataset: {len(image_data)}")
print('Most Common Pokemon: ', Counter(labels).most_common(1)[0])
print('Least Common Pokemon: ', Counter(labels).most_common()[-1])
#print(f"Labels: {sorted(set(labels))}") # using set removes duplicates / only shows each label once as they appear - just a reminder of all the pokemon in the dataset

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels) # this converts our labels from names to numbers: ['Abra', 'Abra', 'Bulbasaur', 'Squirtle', 'Zubat'] to [0, 0, 1, 2, 3] (if we just had 4 unique pokemon/labels and 5 samples (2 Abras))
onehot_encoded_labels = to_categorical(integer_encoded) # now this converts [0, 0, 1, 2, 3] to [ [1,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1] ] - now the length is just 4, the number of unique pokemon, and we 'one-hot-encode' the data to return 1's in each element in the i'th subvector corresponding to matches in 'labels' for the i'th pokemon

# Split data - use 80% to train the model on, and 20% unseen test data to test the model on
X_train, X_test, y_train, y_test = train_test_split(image_data, onehot_encoded_labels, test_size=0.2, random_state=68)

# Define the model - at the moment I am just using generic layers
act = 0
if act == 1:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64,64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(set(labels)), activation='softmax')
    ])

from tensorflow import keras

model = Sequential(name='RGBimg_Classify_Net')
model.add(keras.layers.Conv2D(128,3,input_shape=(256,256, 3),activation='relu'))
model.add(keras.layers.MaxPool2D())
#model.add(keras.layers.Conv2D(128,3,activation='relu'))
#model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Conv2D(128,3,strides=(2,2),activation='relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64,3,strides=(2,2),activation='relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1024,activation='relu'))
model.add(keras.layers.Dense(512,activation='relu'))
model.add(keras.layers.Dense(len(set(labels)),activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # mocel.compile 'seals' the model - 'adam' is a built in optimiser with learning rate defined at 0.001

history = model.fit(X_train, y_train, epochs=20, validation_split=0.1, batch_size=8) # this gives us a log of how the model parameters, loss, accuracy etc. are changing with each iteration (well epoch really)

# note that just having called model.fit() causes parameters to be iteratively updated in this scope and the final model is stored now - dont need a separate line to say 'now lets start fitting'

test_loss, test_acc = model.evaluate(X_test, y_test) # now lets test how well the model worked on unseen data
print(f"Test accuracy: {test_acc}")

#-------------------------------------------------

# Now I'd like to visually see an example of the model working on some test sample- display a random image from X_test with its true and predicted label

random_idx = np.random.randint(0, len(X_test))
selected_image = X_test[random_idx]
true_label_idx = np.argmax(y_test[random_idx]) # in the end we have a matrix of probabilities per pokemon - argmax selects pokemon with highest predicted probability

predictions = model.predict(np.expand_dims(selected_image, axis=0)) # model.predict calls the final iteration of the model generated earlier with model.fit - also we slightly modify the dimensions of selected image data so it fits with model.predict expectations (expects a batch, so we input batch of size 1)
predicted_label_index = np.argmax(predictions) # returns pokemon associated with highest (argmax) probability

predicted_label = label_encoder.inverse_transform([predicted_label_index])[0] # converting label index back to label name eg 0 -> 'Abra'
true_label = label_encoder.inverse_transform([true_label_idx])[0]

plt.imshow(selected_image) # naturally takes 0 to 1 floats as input to show image so we dont have to rescale RGB values
plt.title(f"True Label: {true_label} | Predicted Label: {predicted_label}")
plt.axis('off')
plt.show()

# --------------------------------------------------

# NEXT STEPS

# Modify/Optimise Epochs, Batch_size, data_split, optimizer, loss, model layers, image resizing, padding, stride, Image.Resampling.BILINEAR

# Currently at 20 epochs we have train_acc = 63%, test_acc = 47%,
# So clearly we are mid accuracy and also overfitting to the training data

# I could convert SVGs to PNG for 13 more samples
