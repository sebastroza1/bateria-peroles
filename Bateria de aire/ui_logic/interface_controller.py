from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from core.camera_manager import CameraManager
from modelos.camshift_model import CamShiftModel
from modelos.iluminacion_model import IluminacionModel

import cv2
import os


class InterfaceController(QMainWindow):
    def __init__(self):
        super().__init__()

        # Cargar UI
        uic.loadUi("ui/Bateria_Interfase.ui", self)

        # Cámara (300x300 fijo)
        self.camera = CameraManager(width=300, height=300)

        # Modelos de seguimiento
        self.models = {
            "Sin modelo": None,
            "Modelo CamShift": CamShiftModel(),
            "Modelo Iluminacion": IluminacionModel()
        }

        # Poblar combo de modelos
        self.Model.clear()
        self.Model.addItems(self.models.keys())
        self.current_model = None
        self.Select_Model.clicked.connect(self.select_model)

        # ---------------------------
        #  Audio (pads + metrónomo)
        # ---------------------------
        self._init_audio()

        # Timer actualización cámara
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_view)
        self.timer.start(30)

        # ---------------------------
        #  PADS DE TAMBORES
        # ---------------------------

        # Líneas superiores (dos)
        self.Y_SUP = 90
        self.LARGO_SUP = 120
        self.MARGEN = 20

        # Línea inferior (una)
        self.Y_INF = 160
        self.LARGO_INF = 140
        self.X_INF = (300 - self.LARGO_INF) // 2

        # Colores velocidad (umbral en px/s)
        self.VEL_BAJA = 80
        self.VEL_ALTA = 190

        # Estado de golpe para evitar repetir sonido
        self.pad1_activo = False
        self.pad2_activo = False
        self.pad3_activo = False

    # ======================================================
    #  Inicialización de audio y combos de sonido
    # ======================================================
    def _init_audio(self):
        # Ruta base de sonidos (../sonidos relativo a este archivo)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.sonidos_base = os.path.join(base_dir, "sonidos")

        # Escanear todos los sonidos disponibles
        todos_los_sonidos = []
        metronomos = []

        if os.path.isdir(self.sonidos_base):
            for root, dirs, files in os.walk(self.sonidos_base):
                for fname in files:
                    if not fname.lower().endswith((".mp3", ".wav", ".ogg")):
                        continue
                    full = os.path.join(root, fname)
                    rel = os.path.relpath(full, self.sonidos_base)

                    # Si pertenece a carpeta metronome -> va al combo Ritmo
                    partes = rel.split(os.sep)
                    if partes[0].lower() == "metronome":
                        metronomos.append(rel)
                    else:
                        todos_los_sonidos.append(rel)

        todos_los_sonidos.sort()
        metronomos.sort()

        # Poblar combos de pads (Sonido1, Sonido2, Sonido3)
        for combo in (self.Sonido1, self.Sonido2, self.Sonido3):
            combo.clear()
            combo.addItem("Sin sonido", "")
            for rel in todos_los_sonidos:
                combo.addItem(rel, rel)

        # Opcional: dejar Sonido4 como extra o vacío
        if hasattr(self, "Sonido4"):
            self.Sonido4.clear()
            self.Sonido4.addItem("Sin sonido", "")
            for rel in todos_los_sonidos:
                self.Sonido4.addItem(rel, rel)

        # Poblar combo Ritmo (metrónomo)
        self.Ritmo.clear()
        self.Ritmo.addItem("Sin metrónomo", "")
        for rel in metronomos:
            self.Ritmo.addItem(rel, rel)

        # Jugadores de audio para cada pad
        self.player_pad1 = QMediaPlayer(self)
        self.player_pad2 = QMediaPlayer(self)
        self.player_pad3 = QMediaPlayer(self)

        # Jugador para metrónomo (aún sin lógica de loop aquí)
        self.player_metronomo = QMediaPlayer(self)

    # ======================================================
    #                      Modelos
    # ======================================================
    def select_model(self):
        option = self.Model.currentText().strip()
        if option not in self.models:
            self.current_model = None
            return

        self.current_model = self.models[option]
        print("Modelo seleccionado:", option)

    # ======================================================
    #                Actualización del frame
    # ======================================================
    def update_camera_view(self):
        frame = self.camera.get_frame()
        if frame is None:
            return

        pos = None
        vel = 0.0

        if self.current_model is not None:
            # Los modelos deben devolver: frame, (x,y), vel
            frame, pos, vel = self.current_model.process(frame)

        # Mostrar texto de posición
        if pos is not None:
            self.Posicion1.setText(f"X: {pos[0]} | Y: {pos[1]}")
        else:
            self.Posicion1.setText("X: - | Y: -")

        # Mostrar texto de velocidad
        self.Posicion2.setText(f"{vel:.2f} px/s")

        # Dibujar tambores y gestionar golpes
        if self.Tambores.isChecked():
            frame = self.draw_tambores(frame, pos, vel)
        else:
            # Si está apagado, reiniciamos estado de golpe
            self.pad1_activo = False
            self.pad2_activo = False
            self.pad3_activo = False

        # Convertir a QImage para mostrar en el QLabel Camara
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.Camara.setPixmap(QPixmap.fromImage(qimg))

    # ======================================================
    #        DIBUJAR LÍNEAS / PADS DE TAMBORES
    # ======================================================
    def draw_tambores(self, frame, pos, vel):
        # Colores base
        verde = (0, 255, 0)
        amarillo = (0, 255, 255)
        rojo = (0, 0, 255)

        # función para decidir color según velocidad
        def color_velocidad():
            if vel < self.VEL_BAJA:
                return amarillo
            elif vel >= self.VEL_ALTA:
                return rojo
            return verde

        # -------------------------------
        # Detectar si hay golpe en cada pad
        # -------------------------------
        golpe_pad1 = False
        golpe_pad2 = False
        golpe_pad3 = False

        if pos is not None:
            x, y = pos

            # Pad superior izquierdo
            if abs(y - self.Y_SUP) < 5 and x < 150:
                golpe_pad1 = True

            # Pad superior derecho
            if abs(y - self.Y_SUP) < 5 and x > 150:
                golpe_pad2 = True

            # Pad inferior central
            if abs(y - self.Y_INF) < 5 and self.X_INF < x < self.X_INF + self.LARGO_INF:
                golpe_pad3 = True

        # =====================================================
        #  PAD SUPERIOR IZQUIERDO
        # =====================================================
        color1 = verde
        if golpe_pad1:
            color1 = color_velocidad()
            if not self.pad1_activo:
                self.play_pad_sound(1, vel)
        self.pad1_activo = golpe_pad1

        x1 = self.MARGEN
        x1_fin = self.MARGEN + self.LARGO_SUP
        cv2.line(frame, (x1, self.Y_SUP), (x1_fin, self.Y_SUP), color1, 4)

        # =====================================================
        #  PAD SUPERIOR DERECHO
        # =====================================================
        color2 = verde
        if golpe_pad2:
            color2 = color_velocidad()
            if not self.pad2_activo:
                self.play_pad_sound(2, vel)
        self.pad2_activo = golpe_pad2

        x2 = 300 - self.MARGEN - self.LARGO_SUP
        x2_fin = x2 + self.LARGO_SUP
        cv2.line(frame, (x2, self.Y_SUP), (x2_fin, self.Y_SUP), color2, 4)

        # =====================================================
        #  PAD CENTRAL INFERIOR
        # =====================================================
        color3 = verde
        if golpe_pad3:
            color3 = color_velocidad()
            if not self.pad3_activo:
                self.play_pad_sound(3, vel)
        self.pad3_activo = golpe_pad3

        cv2.line(
            frame,
            (self.X_INF, self.Y_INF),
            (self.X_INF + self.LARGO_INF, self.Y_INF),
            color3,
            4,
        )

        return frame

    # ======================================================
    #   Reproducir sonido para cada pad con volumen según vel
    # ======================================================
    def play_pad_sound(self, pad_index: int, vel: float):
        # Seleccionar combo y player
        if pad_index == 1:
            combo = self.Sonido1
            player = self.player_pad1
        elif pad_index == 2:
            combo = self.Sonido2
            player = self.player_pad2
        elif pad_index == 3:
            combo = self.Sonido3
            player = self.player_pad3
        else:
            return

        rel_path = combo.currentData()
        if not rel_path:
            return  # "Sin sonido"

        full_path = os.path.join(self.sonidos_base, rel_path)
        if not os.path.isfile(full_path):
            print("No se encontró el archivo de sonido:", full_path)
            return

        url = QUrl.fromLocalFile(full_path)
        player.setMedia(QMediaContent(url))

        # Mapear velocidad a volumen (0–100)
        # Golpe suave -> volumen bajo, golpe rápido -> volumen alto
        if vel <= 0:
            volume = 30
        else:
            # Normalizamos tomando VEL_ALTA como referencia
            ratio = min(vel / float(self.VEL_ALTA), 2.0)  # hasta 2x el umbral alta
            volume = int(30 + (70 * min(ratio, 1.0)))  # entre 30 y 100

        player.setVolume(volume)
        player.stop()
        player.play()

    # ======================================================
    #  Cerrar ventana
    # ======================================================
    def closeEvent(self, event):
        self.camera.release()
        event.accept()
