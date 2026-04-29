# =============================================================================
# TestingRealtimeModel.py
# =============================================================================
# Tujuan:
#   Menjalankan deteksi bahasa isyarat SIBI secara realtime menggunakan model
#   AI (BiLSTM + Attention + Focal Loss + Label Smoothing) yang telah dilatih
#   di Training11Class.ipynb (versi v2).
#
#   Fitur utama:
#     - Deteksi realtime berbasis urutan frame keypoints
#     - Mekanisme stabilisasi: prediksi hanya diterima jika konsisten selama
#       12 frame berturut-turut (mencegah hasil "gonta-ganti")
#     - Threshold kepercayaan minimum 80% sebelum kata diterima
#     - Mode mirror (flip horizontal) yang bisa di-toggle dengan tombol 'M'
#     - Visualisasi bar probabilitas per kelas di layar
#     - Overlay top-right: top-N kelas dengan persentase terkini (compact)
#     - Auto-reset buffer saat tangan tidak terdeteksi
#
# Cara Penggunaan:
#   Pastikan file model tersedia di MODEL_PATH, lalu:
#   python TestingRealtimeModel.py
#   Kontrol keyboard: 'Q' = keluar, 'M' = toggle mirror
#
# PENTING - Konsistensi dengan Dataset Collector & Training v2:
#   - Resolusi kamera     : 1280x720 (sama dengan pengambilan dataset)
#   - Fitur per frame     : 258 (pose=132, lh=63, rh=63) — TANPA face landmark
#   - refine_face_landmarks: False
#   - sequence_length     : 45 frame (sama dengan training)
#
# CATATAN: Karena model v2 sudah belajar mirror via aug_mirror_hands,
#   secara teknis mirror_mode TIDAK lagi krusial untuk akurasi prediksi.
#   Tetap dipertahankan demi UX (user lebih nyaman lihat tampilan mirror).
# =============================================================================

# ==================== IMPORT ====================
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import keras
from keras.layers import Layer
import os
import math
from Config import sequence_length

# ==================== CUSTOM CLASSES (HARUS sama persis dengan training v2) ====================

@keras.saving.register_keras_serializable()
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()


# ── Focal loss versi BARU yang dipakai di Training11Class.ipynb (v2) ─────
# Class subclass tf.keras.losses.Loss agar tersimpan di model file dan
# bisa di-load TANPA perlu meneruskan custom_objects yang rumit.
@tf.keras.utils.register_keras_serializable(package='custom')
class FocalLossWithSmoothing(tf.keras.losses.Loss):
    """Focal loss + label smoothing — IDENTIK dengan yang dipakai saat training."""
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.05,
                 name='focal_loss_with_smoothing', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else [1.0] * 11
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        n_classes = tf.shape(y_pred)[-1]
        y_true_smooth = y_true * (1.0 - self.label_smoothing) + \
                        self.label_smoothing / tf.cast(n_classes, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true_smooth * tf.math.log(y_pred)
        alpha_tensor = tf.constant(self.alpha, dtype=tf.float32)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        alpha_w = tf.reduce_sum(y_true * alpha_tensor, axis=-1, keepdims=True)
        focal_w = alpha_w * tf.pow(1.0 - pt, self.gamma)
        return tf.reduce_mean(focal_w * ce)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'label_smoothing': self.label_smoothing,
        })
        return cfg


# ── Backwards-compat: kalau model lama (5 kelas) masih perlu di-load ─────
def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss versi lama, hanya untuk fallback model legacy."""
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce     = -y_true * tf.math.log(y_pred)
        pt     = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        return tf.reduce_mean(alpha * tf.pow(1.0 - pt, gamma) * ce)
    return loss_fn


# ==================== CONFIGURATION ====================
label_3_class  = np.array(['Saya', 'Makan', 'Obat'])
label_5_class  = np.array(['Saya', 'Makan', 'Obat', 'Agar', 'Kuat'])
label_11_class = np.array(['Saya', 'Makan', 'Obat', 'Agar', 'Kuat',
                           'Buah', 'Sayur', 'Ibu', 'An', 'Sabar', 'Siap'])

ACTIONS   = label_11_class
COLORS = [
    (245, 117,  16),   # 1  - Orange
    (117, 245,  16),   # 2  - Hijau
    ( 16, 117, 245),   # 3  - Biru
    (245,  16, 117),   # 4  - Pink
    ( 16, 245, 245),   # 5  - Cyan
    (245, 245,  16),   # 6  - Kuning
    (180,  16, 245),   # 7  - Ungu
    (245,  16,  16),   # 8  - Merah
    ( 16, 245, 117),   # 9  - Hijau Muda
    (117,  16, 245),   # 10 - Indigo
    (245, 180,  16),   # 11 - Kuning Tua
    ( 16, 245, 180),   # 12 - Tosca
    (245,  16, 180),   # 13 - Magenta
    ( 16, 180, 245),   # 14 - Biru Muda
    (180, 245,  16),   # 15 - Lime
    (245, 100, 100),   # 16 - Salmon
    (100, 245, 200),   # 17 - Mint
    (200, 100, 245),   # 18 - Lavender
    (245, 200, 100),   # 19 - Peach
    (100, 200, 245),   # 20 - Sky Blue
]
THRESHOLD = 0.80   # Akurasi minimal 80%
# ✅ DIPERBAIKI: ganti path sesuai output Training11Class.ipynb v2
# (folder 'Epoch300' sesuai PATH_CONFIGURATION_GENERAL di notebook)
MODEL_PATH = 'TrainingModel/Training11Class/Epoch250/model11class.keras'

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model '{MODEL_PATH}' tidak ditemukan!")
    exit()

# ✅ DIPERBAIKI: load model dengan custom_objects yang BENAR untuk training v2
#    Sebelumnya pakai 'loss_fn': focal_loss() → tidak match dengan model baru
#    yang loss-nya FocalLossWithSmoothing (subclass Loss).
print("🔄 Memuat Model AI...")
try:
    # Karena AttentionLayer & FocalLossWithSmoothing sudah pakai
    # @register_keras_serializable, sebenarnya custom_objects opsional —
    # tapi tetap di-pass demi keamanan jika decorator belum ter-register.
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'AttentionLayer': AttentionLayer,
            'FocalLossWithSmoothing': FocalLossWithSmoothing,
        },
        compile=False,   # ← skip compile saat inference, lebih aman & cepat
    )
    print(f"✅ Model Siap! Shape input  : {model.input_shape}")
    print(f"             Shape output : {model.output_shape}")
    # Validasi: jumlah kelas output harus = jumlah ACTIONS
    n_out = model.output_shape[-1]
    if n_out != len(ACTIONS):
        print(f"⚠️  PERINGATAN: model output ({n_out}) ≠ jumlah ACTIONS ({len(ACTIONS)})")
        print("   Pastikan ACTIONS sesuai dengan kelas yang dilatih.")
except Exception as e:
    # Fallback: coba load dengan focal_loss versi lama (untuk model legacy)
    print(f"⚠️  Gagal load dengan FocalLossWithSmoothing: {e}")
    print("    Mencoba fallback ke focal_loss lama...")
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'AttentionLayer': AttentionLayer,
                'loss_fn': focal_loss(),
            },
            compile=False,
        )
        print("✅ Model legacy berhasil di-load (compile=False).")
    except Exception as e2:
        print(f"❌ Gagal total: {e2}")
        exit()

print("Tekan 'M' untuk Mirror, 'Q' untuk Keluar.")

# ==================== MEDIAPIPE SETUP ====================
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

# ==================== FUNCTIONS ====================

def mediapipe_detection(image, model, is_mirrored):
    if is_mirrored:
        image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    """
    Gambar landmark POSE, TANGAN KIRI, TANGAN KANAN.
    FACE LANDMARK TIDAK DIGAMBAR — konsisten dengan pengambilan dataset.
    """
    # Pose
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10),   thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 44, 121),  thickness=1, circle_radius=1))
    # Tangan Kiri
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76),  thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    # Tangan Kanan
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


def extract_keypoints(results):
    """
    Ekstrak keypoints TANPA face landmark — identik dengan dataset collector.
    Total fitur: pose(33*4=132) + left_hand(21*3=63) + right_hand(21*3=63) = 258
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 4)

    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])  # shape: (258,)


def kalibrasi_focal_length(image, results, jarak_nyata_cm=60):
    """
    Hitung focal length kamera secara otomatis.
    User berdiri tepat di jarak_nyata_cm dari kamera saat kalibrasi.
    Referensi: lebar bahu (pose landmark 11=kiri, 12=kanan)
    """
    if results.pose_landmarks:
        h, w, _ = image.shape
        bl = results.pose_landmarks.landmark[11]
        br = results.pose_landmarks.landmark[12]

        x1, y1 = bl.x * w, bl.y * h
        x2, y2 = br.x * w, br.y * h
        w_pixel = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if w_pixel < 10:
            return None

        W_BAHU = 40  # lebar bahu rata-rata ~40 cm
        return (w_pixel * jarak_nyata_cm) / W_BAHU

    return None


def hitung_jarak_bahu(image, results, focal_length):
    """Estimasi jarak user ke kamera berdasarkan lebar bahu (cm)."""
    if results.pose_landmarks and focal_length:
        h, w, _ = image.shape
        bl = results.pose_landmarks.landmark[11]
        br = results.pose_landmarks.landmark[12]

        x1, y1 = bl.x * w, bl.y * h
        x2, y2 = br.x * w, br.y * h
        w_pixel = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if w_pixel == 0:
            return 0

        W_BAHU = 40
        return int((W_BAHU * focal_length) / w_pixel)

    return 0


def tampilkan_info_jarak(image, jarak_cm, y_offset=30):
    """
    Tampilkan indikator jarak dengan:
    - Angka jarak (cm)
    - Bar horizontal berwarna
    - Pesan panduan
    """
    JARAK_MIN = 50
    JARAK_MAX = 100
    BAR_X     = 20
    BAR_Y     = y_offset + 45
    BAR_W     = 260
    BAR_H     = 14

    if jarak_cm == 0:
        warna = (128, 128, 128)
        pesan = "Bahu tidak terdeteksi"
        persen = 0
    elif jarak_cm < JARAK_MIN:
        warna = (0, 60, 255)
        pesan = "<< MUNDUR!"
        persen = max(0, jarak_cm / JARAK_MIN)
    elif jarak_cm > JARAK_MAX:
        warna = (0, 60, 255)
        pesan = "MAJU! >>"
        persen = 1.0
    else:
        warna = (0, 210, 0)
        pesan = "POSISI OK"
        persen = (jarak_cm - JARAK_MIN) / (JARAK_MAX - JARAK_MIN)

    cv2.putText(image, f"Jarak: {jarak_cm} cm", (BAR_X, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, warna, 2)
    cv2.putText(image, pesan, (BAR_X + 155, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, warna, 1)

    cv2.rectangle(image, (BAR_X, BAR_Y), (BAR_X + BAR_W, BAR_Y + BAR_H),
                  (50, 80, 50), -1)
    cv2.rectangle(image, (BAR_X, BAR_Y), (BAR_X + BAR_W, BAR_Y + BAR_H),
                  (180, 180, 180), 1)
    fill_w = int(BAR_W * min(persen, 1.0))
    if fill_w > 0:
        cv2.rectangle(image, (BAR_X, BAR_Y),
                      (BAR_X + fill_w, BAR_Y + BAR_H), warna, -1)

    cv2.putText(image, f"{JARAK_MIN}cm", (BAR_X, BAR_Y + BAR_H + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    cv2.putText(image, f"{JARAK_MAX}cm",
                (BAR_X + BAR_W - 38, BAR_Y + BAR_H + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        color = colors[num] if num < len(colors) else (255, 255, 255)
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), color, -1)
        text = f"{actions[num]}: {prob * 100:.2f}%"
        cv2.putText(output_frame, text, (5, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return output_frame


def draw_top_right_overlay(image, res, actions, colors, top_n=5, is_locked=False):
    """
    Tampilkan overlay compact Top-N kelas dengan persentase di pojok kanan atas.
    """
    h, w = image.shape[:2]

    ROW_H     = 26
    PADDING   = 8
    PANEL_W   = 230
    BAR_MAX_W = 70
    MARGIN_R  = 10
    MARGIN_T  = 50

    BORDER_COLOR = (0, 210, 80) if is_locked else (100, 100, 100)

    top_indices = np.argsort(res)[::-1][:top_n]
    n_rows      = len(top_indices)

    panel_h = PADDING * 2 + ROW_H * n_rows + 22
    panel_x = w - PANEL_W - MARGIN_R
    panel_y = MARGIN_T

    bg_color = (10, 30, 10) if is_locked else (20, 20, 20)
    overlay  = image.copy()
    cv2.rectangle(overlay,
                  (panel_x, panel_y),
                  (panel_x + PANEL_W, panel_y + panel_h),
                  bg_color, -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    border_thick = 2 if is_locked else 1
    cv2.rectangle(image,
                  (panel_x, panel_y),
                  (panel_x + PANEL_W, panel_y + panel_h),
                  BORDER_COLOR, border_thick)

    if is_locked:
        lx, ly = panel_x + PADDING, panel_y + 6
        cv2.rectangle(image, (lx, ly + 3), (lx + 8, ly + 9), (0, 210, 80), -1)
        cv2.rectangle(image, (lx + 2, ly), (lx + 6, ly + 5), (0, 210, 80), 1)
        cv2.putText(image, "TERKUNCI",
                    (panel_x + PADDING + 12, panel_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 210, 80), 1, cv2.LINE_AA)
    else:
        cv2.putText(image, "Top Prediksi",
                    (panel_x + PADDING, panel_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.line(image,
             (panel_x + PADDING, panel_y + 20),
             (panel_x + PANEL_W - PADDING, panel_y + 20),
             BORDER_COLOR if is_locked else (80, 80, 80), 1)

    for rank, idx in enumerate(top_indices):
        prob  = float(res[idx])
        label = actions[idx] if idx < len(actions) else f"Class {idx}"
        color = colors[idx] if idx < len(colors) else (200, 200, 200)

        row_y  = panel_y + 20 + PADDING + rank * ROW_H
        text_y = row_y + ROW_H - 9

        if rank == 0:
            hi = image.copy()
            hi_color = (20, 60, 20) if is_locked else (50, 50, 50)
            cv2.rectangle(hi,
                          (panel_x + 2, row_y + 2),
                          (panel_x + PANEL_W - 2, row_y + ROW_H - 1),
                          hi_color, -1)
            cv2.addWeighted(hi, 0.5, image, 0.5, 0, image)

        BULLET_X = panel_x + PADDING
        cv2.circle(image, (BULLET_X + 4, text_y - 4), 4, color, -1)

        label_display = label if len(label) <= 10 else label[:9] + "."
        label_color   = (255, 255, 255) if rank == 0 else (180, 180, 180)
        cv2.putText(image, label_display,
                    (BULLET_X + 14, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1, cv2.LINE_AA)

        BAR_X = panel_x + PADDING + 95
        BAR_Y = row_y + 7
        BAR_H = 8
        cv2.rectangle(image, (BAR_X, BAR_Y), (BAR_X + BAR_MAX_W, BAR_Y + BAR_H),
                      (50, 50, 50), -1)
        fill = int(BAR_MAX_W * prob)
        if fill > 0:
            bar_color = (0, 210, 80) if (is_locked and rank == 0) else color
            cv2.rectangle(image, (BAR_X, BAR_Y), (BAR_X + fill, BAR_Y + BAR_H),
                          bar_color, -1)

        pct_text  = f"{prob * 100:5.1f}%"
        if is_locked and rank == 0:
            pct_color = (80, 255, 80)
        elif not is_locked and rank == 0 and prob >= THRESHOLD:
            pct_color = (0, 255, 255)
        else:
            pct_color = (160, 160, 160)
        cv2.putText(image, pct_text,
                    (BAR_X + BAR_MAX_W + 4, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, pct_color, 1, cv2.LINE_AA)

    return image


# ==================== REALTIME LOGIC ====================
sequence    = []
sentence    = []
predictions = []
mirror_mode = True

STABILITY_FRAMES = 12
JARAK_KALIBRASI  = 60   # cm — jarak saat kalibrasi bahu
focal_length     = None

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=False,
        enable_segmentation=False) as holistic:

    # -----------------------------------------------
    # PHASE 0: KALIBRASI FOCAL LENGTH
    # -----------------------------------------------
    print(f"\n[KALIBRASI] Berdiri tepat {JARAK_KALIBRASI}cm dari kamera, lalu tekan SPASI.")
    print("Tekan [M] untuk toggle mirror, [Q] untuk keluar.")

    while focal_length is None:
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic, mirror_mode)
        draw_styled_landmarks(image, results)

        h, w = image.shape[:2]

        cv2.rectangle(image, (0, 0), (w, 105), (0, 0, 0), -1)
        cv2.putText(image, "FASE KALIBRASI", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        cv2.putText(image, f"Berdiri tepat {JARAK_KALIBRASI}cm dari kamera",
                    (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, "SPASI = kalibrasi  |  M = mirror  |  Q = keluar",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        bahu_ok    = results.pose_landmarks is not None
        st_bahu    = "Bahu terdeteksi OK" if bahu_ok else "Bahu belum terdeteksi..."
        warna_bahu = (0, 220, 0) if bahu_ok else (0, 0, 220)
        cv2.putText(image, st_bahu, (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_bahu, 2)

        mode_text = "M: ON" if mirror_mode else "M: OFF"
        cv2.putText(image, mode_text, (w - 90, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('SIBI Test', image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if bahu_ok:
                fl = kalibrasi_focal_length(image, results, JARAK_KALIBRASI)
                if fl:
                    focal_length = fl
                    print(f"Kalibrasi berhasil! Focal length: {focal_length:.1f}px")
                else:
                    print("Kalibrasi gagal, lebar bahu terlalu kecil. Coba lagi.")
            else:
                print("Bahu tidak terdeteksi.")
        elif key == ord('m'):
            mirror_mode = not mirror_mode
            sequence = []
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # -----------------------------------------------
    # PHASE 1: DETEKSI REALTIME
    # -----------------------------------------------
    last_res = None
    locked_res     = None
    overlay_locked = False
    prev_sentence_len = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic, mirror_mode)
        draw_styled_landmarks(image, results)

        jarak_cm = hitung_jarak_bahu(image, results, focal_length)

        hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

        if hand_detected:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]
        else:
            sequence       = []
            predictions    = []
            overlay_locked = False

        live_status = "Menunggu Tangan..." if not hand_detected else "Menganalisa..."
        live_prob   = 0.0
        text_color  = (0, 0, 255)

        if len(sequence) == sequence_length:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            last_res = res

            best_class_index = np.argmax(res)
            confidence       = res[best_class_index]

            predictions.append(best_class_index)
            recent_predictions = predictions[-STABILITY_FRAMES:]

            live_status = ACTIONS[best_class_index]
            live_prob   = confidence

            word_accepted = False
            if (len(recent_predictions) == STABILITY_FRAMES
                    and len(np.unique(recent_predictions)) == 1
                    and np.unique(recent_predictions)[0] == best_class_index
                    and confidence > THRESHOLD
                    and not overlay_locked):
                sentence.append(ACTIONS[best_class_index])
                word_accepted = True

            if len(sentence) > 5:
                sentence = sentence[-5:]

            if word_accepted:
                locked_res     = res.copy()
                overlay_locked = True
                predictions    = []

            image = prob_viz(res, ACTIONS, image, COLORS)

            if confidence > THRESHOLD:
                text_color = (0, 255, 0)

        display_res = locked_res if overlay_locked else last_res
        if display_res is not None:
            image = draw_top_right_overlay(image, display_res, ACTIONS, COLORS,
                                           top_n=5, is_locked=overlay_locked)

        h, w = image.shape[:2]

        cv2.rectangle(image, (0, 0), (w, 40), (245, 117, 16), -1)
        cv2.putText(image, 'Kalimat: ' + ' '.join(sentence), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if hand_detected:
            display_text = f"Deteksi: {live_status} ({live_prob * 100:.1f}%)"
        else:
            display_text = "TANGAN TIDAK TERDETEKSI"

        cv2.putText(image, display_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        mode_text = "M: ON" if mirror_mode else "M: OFF"
        cv2.putText(image, mode_text, (w - 90, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        tampilkan_info_jarak(image, jarak_cm, y_offset=530)

        cv2.imshow('SIBI Test', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            mirror_mode = not mirror_mode
            sequence = []

cap.release()
cv2.destroyAllWindows()