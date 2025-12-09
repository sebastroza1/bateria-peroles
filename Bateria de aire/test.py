import cv2
import numpy as np
import time

# ============================================================
# DETECTOR DE DESTELLOS (BRILLO LOCAL)
# ============================================================

class DestelloDetector:

    def __init__(self):
        self.kernel = np.ones((5, 5), np.uint8)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar regiones muy brillantes
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, mask

        # Tomar el destello más grande
        cnt = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w//2, y + h//2

        return (cx, cy), mask


# ============================================================
# DETECCIÓN Y AJUSTE DINÁMICO DE COLOR AZUL
# ============================================================

class ColorAzulDinamico:

    def __init__(self):
        # Valores iniciales amplios
        self.lower_blue = np.array([90, 40, 40])
        self.upper_blue = np.array([130, 255, 255])

        self.kernel = np.ones((3, 3), np.uint8)

    def ajustar_hsv(self, roi_hsv):
        if roi_hsv.size == 0:
            return

        h_mean = np.mean(roi_hsv[:, :, 0])
        s_mean = np.mean(roi_hsv[:, :, 1])
        v_mean = np.mean(roi_hsv[:, :, 2])

        h_range, s_range, v_range = 15, 80, 80

        # Ajustar rangos dinámicamente
        self.lower_blue = np.array([
            max(h_mean - h_range, 0),
            max(s_mean - s_range, 0),
            max(v_mean - v_range, 0)
        ], dtype=np.uint8)

        self.upper_blue = np.array([
            min(h_mean + h_range, 179),
            min(s_mean + s_range, 255),
            min(v_mean + v_range, 255)
        ], dtype=np.uint8)

    def detectar_esfera(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_raw = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        mask = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        best_score = 0
        stats = {"area": 0, "circularidad": 0}

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > 4000:
                continue

            per = cv2.arcLength(cnt, True)
            if per == 0: 
                continue

            circ = 4 * np.pi * area / (per * per)

            if circ > 0.45 and circ > best_score:
                best_cnt = cnt
                best_score = circ
                stats = {"area": area, "circularidad": circ}

        return best_cnt, mask_raw, mask, stats


# ============================================================
# CAMSHIFT PARA SEGUIMIENTO
# ============================================================

class CamShiftTracker:

    def __init__(self):
        self.roi_hist = None
        self.track_window = None

        self.prev_center = None
        self.prev_time = None

    def init(self, frame, cnt):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        x, y, w, h = cv2.boundingRect(cnt)
        self.track_window = (x, y, w, h)

        roi = hsv[y:y+h, x:x+w]
        mask = cv2.inRange(roi, np.array([0, 40, 40]), np.array([179, 255, 255]))

        hist = cv2.calcHist([roi], [0,1], mask, [180,256], [0,180,0,256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        self.roi_hist = hist

    def track(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backproj = cv2.calcBackProject([hsv], [0,1], self.roi_hist, [0,180,0,256], 1)

        ret, self.track_window = cv2.CamShift(
            backproj,
            self.track_window,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        )

        pts = np.intp(cv2.boxPoints(ret))
        cv2.polylines(frame, [pts], True, (0,255,0), 2)

        cx, cy = map(int, ret[0])

        now = time.time()
        vel = 0

        if self.prev_center is not None:
            dx = cx - self.prev_center[0]
            dy = cy - self.prev_center[1]
            dist = np.sqrt(dx*dx + dy*dy)
            dt = now - self.prev_time
            if dt > 0:
                vel = dist / dt

        self.prev_center = (cx, cy)
        self.prev_time = now

        return frame, (cx, cy), vel, backproj


# ============================================================
# PIPELINE COMPLETO
# ============================================================

def main():
    cap = cv2.VideoCapture(0)

    destello_detector = DestelloDetector()
    color_dinamico = ColorAzulDinamico()
    tracker = CamShiftTracker()

    camshift_activo = False
    perdida_frames = 0
    TARGET = 300

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W = frame.shape[:2]
        frame_proc = cv2.resize(frame, (TARGET, TARGET))
        sx, sy = W / TARGET, H / TARGET

        # 1️⃣ DETECCIÓN DE DESTELLO
        destello, mask_destello = destello_detector.detect(frame_proc)

        # 2️⃣ SI HAY DESTELLO, TOMAR ROI PARA AJUSTAR COLOR DINÁMICO
        if destello:
            cx, cy = destello
            x1, y1 = max(cx-20,0), max(cy-20,0)
            x2, y2 = min(cx+20,TARGET), min(cy+20,TARGET)

            roi = frame_proc[y1:y2, x1:x2]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            color_dinamico.ajustar_hsv(hsv_roi)

        # 3️⃣ DETECTAR ESFERA USANDO COLOR DINÁMICO
        cnt, mask_raw, mask_blue, stats = color_dinamico.detectar_esfera(frame_proc)

        # 4️⃣ ACTIVAR CAMSHIFT SI HAY ESFERA VÁLIDA
        if not camshift_activo and cnt is not None:
            tracker.init(frame_proc, cnt)
            camshift_activo = True

        # 5️⃣ SEGUIMIENTO CAMSHIFT
        estado = "Detectando..."
        vel = 0

        if camshift_activo:
            frame_proc, center, vel, backproj = tracker.track(frame_proc)
            estado = f"Tracking | Vel={vel:.1f}px/s"

            if cnt is None:
                perdida_frames += 1
            else:
                perdida_frames = 0

            if perdida_frames > 10:
                camshift_activo = False
                perdida_frames = 0
                estado = "Perdido → Reinicio..."

        # 6️⃣ CONVERTIR COORDENADAS A IMAGEN REAL
        if camshift_activo:
            cx_full = int(center[0] * sx)
            cy_full = int(center[1] * sy)
            cv2.circle(frame, (cx_full, cy_full), 10, (255,0,0), -1)

        # 7️⃣ MOSTRAR INFO
        cv2.putText(frame, f"Area: {stats['area']:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)
        cv2.putText(frame, f"Circ: {stats['circularidad']:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)
        cv2.putText(frame, estado, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)

        # 8️⃣ MOSTRAR TODO
        cv2.imshow("Tracker combinado", frame)
        cv2.imshow("Destellos", mask_destello)
        cv2.imshow("Mascara Azul RAW", mask_raw)
        cv2.imshow("Mascara Azul Filtrada", mask_blue)

        if camshift_activo:
            cv2.imshow("BackProjection", backproj)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
