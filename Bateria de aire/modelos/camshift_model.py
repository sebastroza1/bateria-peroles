import cv2
import numpy as np
import time

class CamShiftModel:

    def __init__(self):
        self.lower_blue = np.array([90, 120, 60])
        self.upper_blue = np.array([130, 255, 255])

        self.kernel = np.ones((5, 5), np.uint8)
        self.roi_hist = None
        self.track_window = None

        self.prev_center = None
        self.prev_time = None

    def choose_circle(self, contours):
        best_cnt = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 60 or area > 5000:
                continue

            per = cv2.arcLength(cnt, True)
            if per == 0:
                continue

            circularidad = 4 * np.pi * area / (per * per)
            if circularidad > best_score:
                best_score = circularidad
                best_cnt = cnt

        return best_cnt

    def process(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask_blue = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, self.kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, self.kernel)

        # -----------------------------
        # Inicializar ROI automática
        # -----------------------------
        if self.roi_hist is None:
            contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = self.choose_circle(contours)

            if cnt is not None:
                x, y, w, h = cv2.boundingRect(cnt)
                self.track_window = (x, y, w, h)

                roi = hsv[y:y+h, x:x+w]
                roi_mask = mask_blue[y:y+h, x:x+w]

                roi_hist = cv2.calcHist([roi], [0, 1], roi_mask,
                                        [180, 256], [0, 180, 0, 256])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                self.roi_hist = roi_hist

            return frame, None, 0

        # -----------------------------
        # Backprojection
        # -----------------------------
        back_proj = cv2.calcBackProject([hsv], [0, 1], self.roi_hist,
                                        [0, 180, 0, 256], 1)
        back_proj = cv2.bitwise_and(back_proj, back_proj, mask=mask_blue)
        back_proj = cv2.GaussianBlur(back_proj, (9, 9), 0)

        # -----------------------------
        # CamShift Tracking
        # -----------------------------
        ret, self.track_window = cv2.CamShift(
            back_proj,
            self.track_window,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        )

        pts = cv2.boxPoints(ret)
        pts = np.intp(pts)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        cx, cy = ret[0]
        cx = int(cx)
        cy = int(cy)

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
