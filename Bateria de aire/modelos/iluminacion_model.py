import cv2
import numpy as np
import time

class IluminacionModel:

    def __init__(self):
        self.lower_blue = np.array([90, 120, 60])
        self.upper_blue = np.array([130, 255, 255])
        self.kernel = np.ones((5, 5), np.uint8)

        self.prev_center = None
        self.prev_time = None

    def process(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20 or area > 2000:
                continue

            per = cv2.arcLength(cnt, True)
            if per == 0:
                continue

            circularidad = 4 * np.pi * area / (per * per)
            if circularidad > 0.4 and circularidad > best_score:
                best_score = circularidad
                best_cnt = cnt

        if best_cnt is None:
            return frame, None, 0

        x, y, w, h = cv2.boundingRect(best_cnt)
        cx = x + w//2
        cy = y + h//2

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)

        # -----------------------------
        # Velocidad
        # -----------------------------
        now = time.time()
        vel = 0

        if self.prev_center is not None:
            dx = cx - self.prev_center[0]
            dy = cy - self.prev_center[1]
            dist = np.sqrt(dx**2 + dy**2)
            dt = now - self.prev_time

            if dt > 0:
                vel = dist / dt

        self.prev_center = (cx, cy)
        self.prev_time = now

        return frame, (cx, cy), vel
