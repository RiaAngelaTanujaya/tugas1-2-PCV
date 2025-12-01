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

# Setup Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Kernel untuk Filter Background
# K_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) # Dihilangkan karena menggunakan background statis

# Pygame & Live2D
pygame.init()
display = (640, 480) # Resolusi jendela Live2D
pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
window_title = "VTuber Tracking App"
pygame.display.set_caption(window_title)
clock = pygame.time.Clock()

live2d.init()
live2d.glewInit()

model_folder = "C:\\Users\\lenovo\\Documents\\PCV\\project2\\model\\runtime"
live2d_model = live2d.LAppModel()
live2d_model.LoadModelJson(os.path.join(model_folder, "mao_pro.model3.json"))
live2d_model.Resize(*display)

# --- GLOBAL SETTINGS ---
STATIC_MODEL = True 
DEFAULT_MODEL_SCALE = 1.5
DEFAULT_MODEL_OFFSET_Y = -0.5

# Smoothing class
class Smoother:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.value = 0.0
    def update(self, new_value):
        if new_value is None: return self.value
        if self.value is None: self.value = new_value
        self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

# Smoothers (Disarankan menggunakan alpha yang lebih rendah untuk lengan agar lebih stabil, tapi saya biarkan sesuai yang Anda berikan di input terakhir)
yaw_smoother = Smoother()
pitch_smoother = Smoother()
roll_smoother = Smoother()
mouth_smoother = Smoother()
eye_l_open_smoother = Smoother(alpha=0.6) 
eye_r_open_smoother = Smoother(alpha=0.6)
body_tilt_smoother = Smoother()
eye_ball_x_smoother = Smoother(alpha=0.2)
eye_ball_y_smoother = Smoother(alpha=0.2)
body_pitch_smoother = Smoother(alpha=0.3)
body_roll_smoother = Smoother(alpha=0.3)
model_pos_x_smoother = Smoother(alpha=0.2)
model_pos_y_smoother = Smoother(alpha=0.2)
model_scale_smoother = Smoother(alpha=0.2)
model_scale_smoother.value = DEFAULT_MODEL_SCALE
model_offset_y_smoother = Smoother(alpha=0.2)
model_offset_y_smoother.value = DEFAULT_MODEL_OFFSET_Y

# Smoother Tangan
shoulder_l_smoother = Smoother(alpha=0.6)
elbow_l_smoother = Smoother(alpha=0.6)
shoulder_r_smoother = Smoother(alpha=0.6)
elbow_r_smoother = Smoother(alpha=0.6)

leg_l_smoother = Smoother(alpha=0.4)
leg_r_smoother = Smoother(alpha=0.4)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera. Pastikan kamera terhubung dan tidak digunakan aplikasi lain.")
    exit()

# --- Variabel Parameter dan Debug ---
param_values = {
    "ParamAngleX": 0.0, "ParamAngleY": 0.0, "ParamAngleZ": 0.0, "ParamA": 0.0,
    "ParamEyeLOpen": 1.0, "ParamEyeROpen": 1.0, "ParamEyeBallX": 0.0, "ParamEyeBallY": 0.0,
    "ParamBodyAngleX": 0.0, "ParamBodyAngleY": 0.0, "ParamBodyAngleZ": 0.0,
    "ParamAllX": 0.0, "ParamAllY": 0.0,
    "ParamArmLA01": 0.0, "ParamArmLA02": 0.0,
    "ParamArmRA01": 0.0, "ParamArmRA02": 0.0,
    "ParamLegL": 0.0,
    "ParamLegR": 0.0,
}
# Variabel debug
# ... (dihilangkan untuk meringkas)

# --- KALIBRASI KONSTANTA ---
EAR_MAX = 0.37
EAR_MIN = 0.15

HAND_Y_MIN_UP = 0.1
HAND_Y_MAX_DOWN = 0.8 

HAND_X_MIN_LEFT = 0.2
HAND_X_MAX_RIGHT = 0.8
KNEE_Y_MIN = 0.5
KNEE_Y_MAX = 0.9
DEFAULT_SHOULDER_Z = -0.5 
ZOOM_SENSITIVITY = 4.0 

# Landmark indices
LEFT_IRIS = 468; LEFT_EYE_CORNERS = [33, 133]; LEFT_EYE_V = [159, 145]
RIGHT_IRIS = 473; RIGHT_EYE_CORNERS = [263, 362]; RIGHT_EYE_V = [386, 374]
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [263, 387, 385, 362, 380, 373]

# --- Fungsi Helper Head Pose (Tidak Diubah) ---
def get_head_pose(landmarks):
    # ... (kode get_head_pose)
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),  
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)   
    ], dtype="double")
    lm_indices = [1, 152, 226, 446, 57, 287]

    try:
        image_points = np.array([
            (landmarks[i].x * display[0], landmarks[i].y * display[1])
            for i in lm_indices if landmarks[i]
        ], dtype="double")
        
        if len(image_points) != len(model_points):
            return 0.0, 0.0, 0.0
    except:
        return 0.0, 0.0, 0.0
    
    focal_length = display[0]
    center = (display[0]/2, display[1]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    dist_coeffs = np.zeros((4,1))
    
    success, rotation_vec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
    
    if success:
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        sy = np.sqrt(rotation_mat[0,0] * rotation_mat[0,0] + rotation_mat[1,0] * rotation_mat[1,0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rotation_mat[2,1], rotation_mat[2,2]) 
            y = np.arctan2(-rotation_mat[2,0], sy)              
            z = np.arctan2(rotation_mat[1,0], rotation_mat[0,0]) 
        else:
            x = np.arctan2(-rotation_mat[1,2], rotation_mat[1,1])
            y = np.arctan2(-rotation_mat[2,0], sy)
            z = 0
        pitch = np.degrees(x)
        yaw = np.degrees(y)
        roll = np.degrees(z)
        yaw = -yaw
        roll = -roll
        pitch = np.clip(pitch, -30, 30)
        yaw = np.clip(yaw, -30, 30)
        roll = np.clip(roll, -30, 30)
        return pitch, yaw, roll
    return 0.0, 0.0, 0.0

# --- Fungsi Helper Mata dan Bola Mata (Tidak Diubah) ---
def get_eye_ear(landmarks, eye_indices):
    # ... (kode get_eye_ear)
    try:
        if not all(landmarks[i] for i in eye_indices): return 0.4
        eye_points = np.array([np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices])
        eye_points[:, 0] *= display[0]
        eye_points[:, 1] *= display[1]
        vertical1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vertical2 = np.linalg.norm(eye_points[2] - eye_points[4])
        horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
        if horizontal == 0: return 0.4
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    except: return 0.4

def get_eye_ball_pos(face_lm, l_iris_idx, l_corner_idx, l_top_idx, l_bottom_idx):
    # ... (kode get_eye_ball_pos)
    try:
        indices_to_check = [l_iris_idx, l_corner_idx[0], l_corner_idx[1], l_top_idx, l_bottom_idx]
        if not all(face_lm[i] for i in indices_to_check): return 0.0, 0.0
        iris_center = np.array([face_lm[l_iris_idx].x * display[0], face_lm[l_iris_idx].y * display[1]])
        eye_corner_left = np.array([face_lm[l_corner_idx[0]].x * display[0], face_lm[l_corner_idx[0]].y * display[1]])
        eye_corner_right = np.array([face_lm[l_corner_idx[1]].x * display[0], face_lm[l_corner_idx[1]].y * display[1]])
        eye_top = np.array([face_lm[l_top_idx].x * display[0], face_lm[l_top_idx].y * display[1]])
        eye_bottom = np.array([face_lm[l_bottom_idx].x * display[0], face_lm[l_bottom_idx].y * display[1]])
        eye_center_x = (eye_corner_left[0] + eye_corner_right[0]) / 2.0
        eye_center_y = (eye_top[1] + eye_bottom[1]) / 2.0
        eye_width = np.linalg.norm(eye_corner_left - eye_corner_right)
        eye_height = np.linalg.norm(eye_top - eye_bottom)
        if eye_width == 0 or eye_height == 0: return 0.0, 0.0
        x_pos = ((iris_center[0] - eye_center_x) / (eye_width / 2.0))
        y_pos = ((iris_center[1] - eye_center_y) / (eye_height / 2.0))
        return np.clip(x_pos, -1.0, 1.0), np.clip(y_pos, -1.0, 1.0)
    except Exception: return 0.0, 0.0


# -----------------------------------------------------------------
# --- FUNGSI BARU: MEMUAT GAMBAR STATIS SEBAGAI BACKGROUND ---
# -----------------------------------------------------------------
def load_static_background(image_path, width, height):
    """Memuat gambar statis dari file ke tekstur OpenGL."""
    try:
        # Load gambar menggunakan OpenCV
        frame_data = cv2.imread(image_path)
        if frame_data is None:
            print(f"Error: Tidak dapat memuat gambar {image_path}. Memuat background hitam.")
            return None, width, height

        # Ubah ukuran gambar agar sesuai dengan resolusi display (640x480)
        frame_data = cv2.resize(frame_data, (width, height), interpolation=cv2.INTER_LINEAR)

        # Konversi BGR ke RGB dan flip vertikal untuk OpenGL
        rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.flip(rgb_frame, 0) 
        
        return rgb_frame, width, height
    except Exception as e:
        print(f"Error saat memuat/memproses gambar background: {e}")
        return None, width, height

# Muat gambar statis sekali sebelum loop utama
BACKGROUND_PATH = "download.jpeg" 
background_rgb_data, bg_w, bg_h = load_static_background(BACKGROUND_PATH, display[0], display[1])
# -----------------------------------------------------------------


# --- Fungsi untuk menggambar background OpenGL (REVISI) ---
def draw_background(frame_texture_id, frame_data, width, height):
    """Menggambar frame (berasal dari gambar statis) sebagai background OpenGL."""
    
    glPushMatrix()
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, width, 0, height, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    if frame_data is not None:
        glBindTexture(GL_TEXTURE_2D, frame_texture_id)
        # Mengganti data tekstur dengan gambar statis yang sudah dimuat
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_data.tobytes())
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, frame_texture_id)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex3f(0.0, 0.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f(width, 0.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f(width, height, 0.0)
        glTexCoord2f(0.0, 1.0); glVertex3f(0.0, height, 0.0)
        glEnd()
        glDisable(GL_TEXTURE_2D)
    else:
        # Fallback ke background hitam jika gambar statis gagal dimuat
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()


# Buat ID tekstur untuk background
frame_texture_id = glGenTextures(1)
glClearColor(0.0, 0.0, 0.0, 1.0)


with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True) as face_mesh, \
      mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose, \
      mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    
    while cap.isOpened():
        # 1. Baca frame kamera (Hanya untuk tracking)
        ret, frame = cap.read()
        if not ret or frame is None:
            rgb_for_tracking = None
        else:
            frame = cv2.flip(frame, 1) # Balik horizontal agar mirror
            rgb_for_tracking = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # 2. Proses frame untuk tracking
        face_results = None
        pose_results = None
        hands_results = None
        
        if rgb_for_tracking is not None:
            rgb_for_tracking.flags.writeable = False
            face_results = face_mesh.process(rgb_for_tracking)
            pose_results = pose.process(rgb_for_tracking)
            hands_results = hands.process(rgb_for_tracking)
            rgb_for_tracking.flags.writeable = True

        # 3. Reset nilai parameter default
        for key in param_values: 
            param_values[key] = 0.0 if "Open" not in key else 1.0 
        
        # --- HILANGKAN SEMUA LOGIKA JENDELA TRACKING NODES ---
        # Semua panggilan mp_drawing.draw_landmarks(...) telah dihapus
        # cv2.imshow("Tracking Nodes", frame_with_nodes) telah dihapus
        # cv2.getWindowProperty("Tracking Nodes", cv2.WND_PROP_VISIBLE) telah dihapus
        
        # --- FACE TRACKING ---
        if face_results and face_results.multi_face_landmarks:
            face_lm = face_results.multi_face_landmarks[0].landmark
            
            # Head Rotation
            pitch_raw, yaw_raw, roll_raw = get_head_pose(face_lm)
            yaw = yaw_smoother.update(yaw_raw); pitch = pitch_smoother.update(pitch_raw); roll = roll_smoother.update(roll_raw)
            live2d_model.SetParameterValue("ParamAngleX", yaw); param_values["ParamAngleX"] = yaw
            live2d_model.SetParameterValue("ParamAngleY", pitch); param_values["ParamAngleY"] = pitch
            live2d_model.SetParameterValue("ParamAngleZ", roll); param_values["ParamAngleZ"] = roll
            
            # Mouth
            mouth_open_raw = abs(face_lm[14].y - face_lm[13].y) * 50
            mouth_open = mouth_smoother.update(mouth_open_raw)
            live2d_model.SetParameterValue("ParamA", mouth_open); param_values["ParamA"] = mouth_open

            # Eye Blink
            left_ear = get_eye_ear(face_lm, LEFT_EYE_INDICES); right_ear = get_eye_ear(face_lm, RIGHT_EYE_INDICES)
            left_open_raw = np.interp(left_ear, [EAR_MIN, EAR_MAX], [0.0, 1.0])
            right_open_raw = np.interp(right_ear, [EAR_MIN, EAR_MAX], [0.0, 1.0])
            left_open = eye_l_open_smoother.update(np.clip(left_open_raw, 0.0, 1.0))
            right_open = eye_r_open_smoother.update(np.clip(right_open_raw, 0.0, 1.0))
            live2d_model.SetParameterValue("ParamEyeLOpen", left_open); param_values["ParamEyeLOpen"] = left_open
            live2d_model.SetParameterValue("ParamEyeROpen", right_open); param_values["ParamEyeROpen"] = right_open

            # Eye Ball
            l_x, l_y = get_eye_ball_pos(face_lm, LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_V[0], LEFT_EYE_V[1])
            r_x, r_y = get_eye_ball_pos(face_lm, RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_V[0], RIGHT_EYE_V[1])
            eye_ball_x = eye_ball_x_smoother.update((l_x + r_x) / 2.0)
            eye_ball_y = eye_ball_y_smoother.update((l_y + r_y) / 2.0)
            live2d_model.SetParameterValue("ParamEyeBallX", eye_ball_x); param_values["ParamEyeBallX"] = eye_ball_x
            live2d_model.SetParameterValue("ParamEyeBallY", eye_ball_y); param_values["ParamEyeBallY"] = eye_ball_y

        # --- POSE/BODY TRACKING ---
        if pose_results and pose_results.pose_landmarks:
            pose_lm = pose_results.pose_landmarks.landmark
            
            if pose_lm[11] and pose_lm[12]: 
                
                # Body Angles & Zoom
                body_tilt_raw = (pose_lm[11].x - pose_lm[12].x) * 150 
                body_tilt = body_tilt_smoother.update(body_tilt_raw)
                live2d_model.SetParameterValue("ParamBodyAngleX", body_tilt); param_values["ParamBodyAngleX"] = body_tilt

                body_roll_raw = (pose_lm[11].y - pose_lm[12].y) * 150 
                body_roll = body_roll_smoother.update(body_roll_raw)
                live2d_model.SetParameterValue("ParamBodyAngleZ", body_roll); param_values["ParamBodyAngleZ"] = body_roll
                
                shoulder_z_avg = (pose_lm[11].z + pose_lm[12].z) / 2.0
                body_pitch_raw = shoulder_z_avg * -50 
                body_pitch = body_pitch_smoother.update(body_pitch_raw)
                live2d_model.SetParameterValue("ParamBodyAngleY", body_pitch); param_values["ParamBodyAngleY"] = body_pitch
                
                # Gerakan 2D Model (ParamAllX, ParamAllY) dan Zoom (SetScale/SetOffset)
                if not STATIC_MODEL:
                    shoulder_x_avg = (pose_lm[11].x + pose_lm[12].x) / 2.0
                    shoulder_y_avg = (pose_lm[11].y + pose_lm[12].y) / 2.0
                    
                    pos_x_raw = np.interp(shoulder_x_avg, [0.2, 0.8], [-1.0, 1.0])
                    pos_x = model_pos_x_smoother.update(np.clip(pos_x_raw, -1.0, 1.0))
                    live2d_model.SetParameterValue("ParamAllX", pos_x); param_values["ParamAllX"] = pos_x
                    
                    pos_y_raw = np.interp(shoulder_y_avg, [0.2, 0.8], [1.0, -1.0])
                    pos_y = model_pos_y_smoother.update(np.clip(pos_y_raw, -1.0, 1.0))
                    live2d_model.SetParameterValue("ParamAllY", pos_y); param_values["ParamAllY"] = pos_y
                    
                    z_diff = DEFAULT_SHOULDER_Z - shoulder_z_avg
                    new_scale_raw = DEFAULT_MODEL_SCALE + (z_diff * ZOOM_SENSITIVITY)
                    new_scale_raw = np.clip(new_scale_raw, 0.8, 3.0) 
                    new_scale = model_scale_smoother.update(new_scale_raw)
                    live2d_model.SetScale(new_scale) 

                    new_offset_y_raw = np.interp(new_scale, [1.0, 1.5, 3.0], [-0.3, -0.5, -1.6])
                    new_offset_y = model_offset_y_smoother.update(new_offset_y_raw)
                    live2d_model.SetOffset(0.0, new_offset_y)
                else:
                    live2d_model.SetParameterValue("ParamAllX", model_pos_x_smoother.update(0.0))
                    live2d_model.SetParameterValue("ParamAllY", model_pos_y_smoother.update(0.0))
                    live2d_model.SetScale(model_scale_smoother.update(DEFAULT_MODEL_SCALE))
                    live2d_model.SetOffset(0.0, model_offset_y_smoother.update(DEFAULT_MODEL_OFFSET_Y))
                    param_values["ParamAllX"] = 0.0 
                    param_values["ParamAllY"] = 0.0 

                
                # Leg Tracking (Lutut)
                if pose_lm[25] and pose_lm[26]:
                    knee_l_y_raw = pose_lm[25].y
                    knee_r_y_raw = pose_lm[26].y
                    
                    val_leg_l_raw = np.interp(knee_l_y_raw, [KNEE_Y_MIN, KNEE_Y_MAX], [10.0, -10.0])
                    val_leg_r_raw = np.interp(knee_r_y_raw, [KNEE_Y_MIN, KNEE_Y_MAX], [10.0, -10.0])
                    
                    val_leg_l = leg_l_smoother.update(np.clip(val_leg_l_raw, -10.0, 10.0))
                    val_leg_r = leg_r_smoother.update(np.clip(val_leg_r_raw, -10.0, 10.0))
                    
                    live2d_model.SetParameterValue("ParamLegL", val_leg_l); param_values["ParamLegL"] = val_leg_l
                    live2d_model.SetParameterValue("ParamLegR", val_leg_r); param_values["ParamLegR"] = val_leg_r

        # --- HAND TRACKING (Masih menggunakan logika lama, tapi ini yang Anda minta) ---
        if hands_results and hands_results.multi_hand_landmarks:
            for idx, hand_lm in enumerate(hands_results.multi_hand_landmarks):
                handedness = hands_results.multi_handedness[idx].classification[0].label
                wrist = hand_lm.landmark[0]
                hand_x = wrist.x; hand_y = wrist.y
                
                param_shoulder_raw = np.interp(hand_y, [HAND_Y_MIN_UP, HAND_Y_MAX_DOWN], [10.0, -10.0]) 
                param_elbow_raw = np.interp(hand_x, [HAND_X_MIN_LEFT, HAND_X_MAX_RIGHT], [-10.0, 10.0])

                param_shoulder = np.clip(param_shoulder_raw, -10.0, 10.0)
                param_elbow = np.clip(param_elbow_raw, -10.0, 10.0)

                if handedness == "Left": 
                    val_shoulder = shoulder_r_smoother.update(param_shoulder)
                    val_elbow = elbow_r_smoother.update(param_elbow)
                    live2d_model.SetParameterValue("ParamArmRA01", val_shoulder)
                    live2d_model.SetParameterValue("ParamArmRA02", val_elbow)
                    param_values["ParamArmRA01"] = val_shoulder
                    param_values["ParamArmRA02"] = val_elbow
                    
                elif handedness == "Right":
                    val_shoulder = shoulder_l_smoother.update(param_shoulder)
                    val_elbow = elbow_l_smoother.update(param_elbow)
                    live2d_model.SetParameterValue("ParamArmLA01", val_shoulder)
                    live2d_model.SetParameterValue("ParamArmLA02", val_elbow)
                    param_values["ParamArmLA01"] = val_shoulder
                    param_values["ParamArmLA02"] = val_elbow

        # --- Print Debug ---
        static_status = "STATIC" if STATIC_MODEL else "DYNAMIC"
        print(f"[{static_status}] Yaw: {param_values['ParamAngleX']:.1f} | Body X: {param_values['ParamBodyAngleX']:.1f} | Pos X: {param_values['ParamAllX']:.1f} | "
              f"L Arm: {param_values['ParamArmLA01']:.1f} | R Arm: {param_values['ParamArmRA01']:.1f} | Leg L: {param_values['ParamLegL']:.1f} | Scale: {model_scale_smoother.value:.2f}")

        # --- Update and render Live2D ---
        # Menggunakan background statis
        draw_background(frame_texture_id, background_rgb_data, bg_w, bg_h)
        live2d_model.Update()
        live2d_model.Draw()
        pygame.display.flip()
        clock.tick(30)

        # cv2.waitKey(1) dihapus karena tidak ada jendela OpenCV yang dibuka
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                cap.release()
                glDeleteTextures(1, [frame_texture_id])
                live2d.dispose()
                pygame.quit()
                cv2.destroyAllWindows()
                exit()
            if event.type == pygame.KEYDOWN:
                # Logika mode filter dihapus karena tidak digunakan
                
                # --- FITUR TOGGLE STATIC MODEL ---
                if event.key == pygame.K_s:
                    STATIC_MODEL = not STATIC_MODEL
                    print(f"*** STATIC MODEL TOGGLED: {'AKTIF' if STATIC_MODEL else 'NONAKTIF'} ***")
                
cap.release()
glDeleteTextures(1, [frame_texture_id])
live2d.dispose()
pygame.quit()
cv2.destroyAllWindows()