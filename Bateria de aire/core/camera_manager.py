import cv2

class CameraManager:
    def __init__(self, camera_index=0, width=300, height=300):
        self.cap = cv2.VideoCapture(camera_index)
        self.width = width
        self.height = height

        # Forzar tamaño fijo
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Redimensionar a 300x300
        frame = cv2.resize(frame, (self.width, self.height))

        # Convertir BGR → RGB (PyQt)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def release(self):
        self.cap.release()
