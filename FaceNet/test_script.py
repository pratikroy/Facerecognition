import cv2
from PIL import Image
from numpy import load
from numpy import expand_dims
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras.models import load_model


class RealTimeFaceDetection:

	def __init__(self):
		self.detector = MTCNN()
		self.video_cap = cv2.VideoCapture(0)
		self.stroke = 1
		self.color = (255, 0, 0)
		print("Loading pre-trained Keras model for face recognition")
		self.keras_model = load_model('facenet_keras.h5', compile=False)
		print("Face recognition model loaded successfully...")

	def img_to_array(self, face_img_pixels, required_size=(160, 160)):
		image = Image.fromarray(face_img_pixels)
		image = image.resize(required_size)
		return asarray(image)

	# Get the face embedding for one face
	def get_embedding(self, model, face_pixels):
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

	def find_faces(self):
		while(self.video_cap.isOpened()):
			ret, frame = self.video_cap.read()
			rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			pixels = asarray(rgb_image)
			all_faces = self.detector.detect_faces(pixels)
			for face in all_faces:
				x1, y1, width, height = face['box']
				x2, y2 = x1 + width, y1 + height
				face_arr = self.img_to_array(pixels[y1:y2, x1:x2])
				face_emb = self.get_embedding(self.keras_model, face_arr)
				print(face_emb.shape)

			cv2.imshow('frame', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				# Release the capture
				self.video_cap.release()
				cv2.destroyAllWindows()


if __name__ == '__main__':
	print("Initializing required parameters...")
	rt_face = RealTimeFaceDetection()
	rt_face.find_faces()