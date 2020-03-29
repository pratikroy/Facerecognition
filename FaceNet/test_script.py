import cv2
import pickle
from PIL import Image
from numpy import load
from numpy import expand_dims
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras.models import load_model

from sklearn.preprocessing import LabelEncoder


class RealTimeFaceDetection:

	def __init__(self):
		self.detector = MTCNN()
		self.video_cap = cv2.VideoCapture(0)
		self.stroke = 1
		self.color = (255, 0, 0)
		print("Loading pre-trained Keras model for face recognition")
		self.keras_model = load_model('facenet_keras.h5', compile=False)
		print("Face recognition model loaded successfully...")
		print("Loading pre-trained SVC model")
		self.svc_model = pickle.load(open('FACENET_MODEL.sav', 'rb'))
		print("Loading successful...")
		self.emb_data = load('5-celebrity-faces-embeddings.npz')

	def img_to_array(self, face_img_pixels, required_size=(160, 160)):
		image = Image.fromarray(face_img_pixels)
		image = image.resize(required_size)
		return asarray(image)

	# Get the face embedding for one face
	def get_embedding(self, model, face_pixels):
		face_pixels = face_pixels.astype('float32')
		mean, std = face_pixels.mean(), face_pixels.std()
		face_pixels = (face_pixels - mean) / std
		samples = expand_dims(face_pixels, axis=0)
		yhat = model.predict(samples)
		return yhat[0]

	def get_encoder(self):
		trainy = self.emb_data['arr_1']
		# Label encode targets
		out_encoder = LabelEncoder()
		out_encoder.fit(trainy)
		return out_encoder

	def find_faces(self):
		out_encoder = self.get_encoder()
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
				samples = expand_dims(face_emb, axis=0)
				yhat_class = self.svc_model.predict(samples)
				predict_names = out_encoder.inverse_transform(yhat_class)
				print(predict_names[0])

			cv2.imshow('frame', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				# Release the capture
				self.video_cap.release()
				cv2.destroyAllWindows()


if __name__ == '__main__':
	print("Initializing required parameters...")
	rt_face = RealTimeFaceDetection()
	rt_face.find_faces()