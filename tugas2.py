import cv2
import numpy as np
import mediapipe as mp
import pygame
import live2d.v3 as live2d
import os
import faulthandler

# --- Import library OpenGL ---
from OpenGL.GL import *
from OpenGL.GLU import *

faulthandler.enable()

# Setup Mediapipe (Minimal setup untuk inisialisasi framework)
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# -----------------------------------------------------------------
# --- KONSTANTA DETEKSI WARNA (BIRU & HIJAU) ---
# -----------------------------------------------------------------
# Warna yang dideteksi: HIJAU
LOWER_GREEN = np.array([35, 50, 50])
UPPER_GREEN = np.array([85, 255, 255])

# Warna yang dideteksi: BIRU
LOWER_BLUE = np.array([90, 50, 50])
UPPER_BLUE = np.array([130, 255, 255])

# Ukuran kernel untuk operasi morfologi (5x5)
MORPH_KERNEL = np.ones((5, 5), np.uint8)

# Konstanta untuk memicu aksi
MIN_CONTOUR_AREA = 5000     # Luas area kontur minimum (piksel)
ARM_TRIGGER_VALUE = 10.0    # Aksi Biru: Gerak Lengan
MOUTH_TRIGGER_VALUE = 50.0  # Aksi Hijau: Buka Mulut (lebih jelas)
NORMAL_ACTION_VALUE = 0.0   # Nilai default

# -----------------------------------------------------------------
# --- LIVE2D & PYGAME SETUP ---
# -----------------------------------------------------------------
pygame.init()
display = (640, 480) 
pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
pygame.display.set_caption("VTuber Color Trigger App (Blue & Green)")
clock = pygame.time.Clock()

live2d.init()
live2d.glewInit()

# !!! GANTI DENGAN PATH MODEL LIVE2D ANDA YANG BENAR !!!
model_folder = "C:\\Users\\lenovo\\Documents\\PCV\\project2\\model\\runtime"
live2d_model = live2d.LAppModel() 
live2d_model.LoadModelJson(os.path.join(model_folder, "mao_pro.model3.json"))
live2d_model.Resize(*display)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

# Variabel Parameter Live2D yang digunakan (Minimal)
param_values = {
    "ParamArmLA01": 0.0, 
    "ParamArmRA01": 0.0, 
    "ParamA": 0.0,      # Mouth Open
}

# Dummy Smoother Class (Agar tidak error jika Live2D memanggilnya)
class Smoother:
    def __init__(self, alpha=0.4): self.value = 0.0
    def update(self, new_value): 
        if new_value is None: return self.value
        self.value = new_value
        return self.value

# -----------------------------------------------------------------
# --- FUNGSI DETEKSI WARNA (TUGAS 2.1, 2.2, 2.3) ---
# -----------------------------------------------------------------
def process_color_detection(frame_bgr):
    """
    Mendeteksi objek berwarna Biru atau Hijau. Prioritas: Hijau > Biru.
    Mengembalikan string warna yang terdeteksi ('GREEN', 'BLUE', atau 'NONE').
    """
    if frame_bgr is None:
        return 'NONE'

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    
    # --- 1. DETEKSI HIJAU (Prioritas Tinggi) ---
    # Thresholding, Opening, dan Closing untuk Hijau
    mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    mask_green_cleaned = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask_green_cleaned = cv2.morphologyEx(mask_green_cleaned, cv2.MORPH_CLOSE, MORPH_KERNEL)
    
    contours_green, _ = cv2.findContours(mask_green_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_green:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            # DEBUG: Gambar kotak hijau
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_bgr, "HIJAU (MULUT)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return 'GREEN' # Hijau terdeteksi, hentikan pengecekan Biru

    # --- 2. DETEKSI BIRU ---
    # Thresholding, Opening, dan Closing untuk Biru
    mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    mask_blue_cleaned = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask_blue_cleaned = cv2.morphologyEx(mask_blue_cleaned, cv2.MORPH_CLOSE, MORPH_KERNEL)

    contours_blue, _ = cv2.findContours(mask_blue_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_blue:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            # DEBUG: Gambar kotak biru
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame_bgr, "BIRU (LENGAN)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            return 'BLUE' # Biru terdeteksi

    return 'NONE' # Tidak ada yang terdeteksi


# -----------------------------------------------------------------
# --- FUNGSI DRAW BACKGROUND ---
# -----------------------------------------------------------------
def draw_background(frame_texture_id, frame_bgr, width, height):
    """Menggambar frame dari kamera sebagai background OpenGL."""
    
    if frame_bgr is None:
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        return

    # Persiapan untuk OpenGL
    if frame_bgr.shape[0] != height or frame_bgr.shape[1] != width:
         frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
         
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.flip(rgb_frame, 0)
    
    glPushMatrix()
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, width, 0, height, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    glBindTexture(GL_TEXTURE_2D, frame_texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_frame.tobytes())
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    glEnable(GL_TEXTURE_2D)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0); glVertex3f(0.0, 0.0, 0.0)
    glTexCoord2f(1.0, 0.0); glVertex3f(width, 0.0, 0.0)
    glTexCoord2f(1.0, 1.0); glVertex3f(width, height, 0.0)
    glTexCoord2f(0.0, 1.0); glVertex3f(0.0, height, 0.0)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

# Buat ID tekstur untuk background
frame_texture_id = glGenTextures(1)
glClearColor(0.0, 0.0, 0.0, 1.0)

# -----------------------------------------------------------------
# --- MAIN LOOP ---
# -----------------------------------------------------------------
# Menggunakan with mp_face_mesh untuk menjaga konsistensi kerangka
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5) as face_mesh: 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            frame_bgr_to_draw = None
        else:
            frame = cv2.flip(frame, 1)
            frame_bgr_to_draw = frame.copy()

        # 1. Deteksi Warna
        detected_color = process_color_detection(frame_bgr_to_draw)

        # 2. Reset dan Atur Aksi Pemicu
        arm_value = NORMAL_ACTION_VALUE
        mouth_value = NORMAL_ACTION_VALUE
        
        if detected_color == 'BLUE':
            # Aksi BIRU: Gerakkan Lengan
            arm_value = ARM_TRIGGER_VALUE
            print(f"*** Aksi Biru: Lengan ke {arm_value} ***")
            
        elif detected_color == 'GREEN':
            # Aksi HIJAU: Buka Mulut
            mouth_value = MOUTH_TRIGGER_VALUE
            print(f"*** Aksi Hijau: Mulut ke {mouth_value} ***")
        
        # 3. Terapkan Aksi ke Model Live2D (TUGAS 2.4)
        live2d_model.SetParameterValue("ParamArmLA01", arm_value)
        live2d_model.SetParameterValue("ParamArmRA01", arm_value)
        live2d_model.SetParameterValue("ParamA", mouth_value)
        
        param_values["ParamArmLA01"] = arm_value
        param_values["ParamArmRA01"] = arm_value
        param_values["ParamA"] = mouth_value
        
        # 4. Render
        draw_background(frame_texture_id, frame_bgr_to_draw, display[0], display[1])
        live2d_model.Update()
        live2d_model.Draw()
        pygame.display.flip()
        clock.tick(30)
        
        # 5. Penanganan Event (Quit/ESC)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                cap.release()
                glDeleteTextures(1, [frame_texture_id])
                live2d.dispose()
                pygame.quit()
                cv2.destroyAllWindows()
                exit()
            
# Cleanup
cap.release()
glDeleteTextures(1, [frame_texture_id])
live2d.dispose()
pygame.quit()
cv2.destroyAllWindows()