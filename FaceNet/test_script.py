import cv2
import sys
from numpy import asarray
from mtcnn.mtcnn import MTCNN


class RealTimeFaceDetection:

	def __init__(self):
		self.detector = MTCNN()
		self.video_cap = cv2.VideoCapture(0)
		self.stroke = 2
		self.color = (255, 0, 0)

	def find_faces(self):
		while(True):
			ret, frame = self.video_cap.read()
			# Convert the image from BGR TO RGB
			rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			pixels = asarray(rgb_image)
			all_faces = self.detector.detect_faces(pixels)
			for face in all_faces:
				x1, y1, width, height = face['box']
				cv2.rectangle(
						frame,
						(x1, y1),
						(x1+width, y1+height),
						self.color,
						self.stroke
					)
			# Diaplay the resulting frame
			cv2.imshow('frame', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				# Release the capture
				self.video_cap.release()
				cv2.destroyAllWindows()


if __name__ == '__main__':
	rt_face = RealTimeFaceDetection()
	rt_face.find_faces()