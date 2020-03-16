from PIL import Image
from os import listdir
from os.path import isdir
from numpy import load
from numpy import asarray
from numpy import expand_dims
from numpy import savez_compressed
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from matplotlib import pyplot
from random import choice

# Import Sklearn for model building
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer



# Extract a single face from a single photograph
def extract_face(filename, required_size=(160,160)):
	# Load image from file
	image = Image.open(filename)
	# Convert to RGB if needed
	image = image.convert('RGB')
	# Convert to array
	pixels = asarray(image)
	# Create the detector using the default weights
	detector = MTCNN()
	# Detect faces in the image
	results = detector.detect_faces(pixels)
	# Extract the bounding box for the face
	x1, y1, width, height = results[0]['box']
	# Make sure to input only positive values
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# Extract the face
	face = pixels[y1:y2, x1:x2]
	# Resize pixels required for model
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array


# Load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	# Enumerate files
	for filename in listdir(directory):
		# Generate the path
		path = directory + filename
		# Get face
		face = extract_face(path)
		# Store them
		faces.append(face)

	return faces


# Load dataset that contains one subdir
def load_dataset(directory):
	X, y = list(), list()
	# Enumerate folders, on per folder
	for subdir in listdir(directory):
		# Generate path
		path = directory + subdir + '/'
		# Skip files
		if not isdir(path):
			continue
		# Load all faces in the sub directory
		faces = load_faces(path)
		# Create labels
		labels = [subdir for _ in range(len(faces))]
		# Summarize progress
		print('loaded %d examples for class: %s' % (len(faces), subdir))
		# Store them
		X.extend(faces)
		y.extend(labels)

	return asarray(X), asarray(y)


# Get the face embedding for one face
def get_embedding(model, face_pixels):
	# Scale pixel values
	face_pixels = face_pixels.astype('float32')
	# Standardize pixel values
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# Transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# Make predictions to get embedding
	yhat = model.predict(samples)

	return yhat[0]


# Load train data set
trainX, trainy = load_dataset('5-celebrity-faces-dataset/train/')
print(trainX.shape, trainy.shape)
# Load test dataset
testX, testy = load_dataset('5-celebrity-faces-dataset/val/')
# save arrays to one file in compressed format
savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)


# Load the face dataset
data = load('5-celebrity-faces-dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print("Loaded: ", trainX.shape, trainy.shape, testX.shape, testy.shape)
# Load the facenet model
model = load_model('facenet_keras.h5', compile=False)
print("Pre-trained face recognition model loaded...")

# Convert training set faces into embedding
newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print("newTrainX shape: ", newTrainX.shape)

# Convert testing set data set into embedding
newTestX = list()
for face_pixels in testX:
	embedding = get_embedding(model, face_pixels)
	newTestX.append(embedding)
newTestX = asarray(newTestX)
print("newTestX shape: ", newTestX.shape)

# Save arrays in compressed format
savez_compressed('5-celebrity-faces-embeddings.npz', newTrainX, trainy, newTestX, testy)


# Used for visualization part
data = load('5-celebrity-faces-dataset.npz')
testX_faces = data['arr_2']
# Load embedding dataset here for measuring accuracy acore
# load dataset
data = load('5-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
# Normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# Label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# Fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# Used for visualization part
# Test model on a random example from the test data set
selection = choice([ i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
# Predict 
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# Compute score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# Print the result
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
# Used for visualization part
# Prediction for the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# Used for visualization part, get the name
class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])
# plot the image for better visualization
pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()