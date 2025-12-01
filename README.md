# Proyek VTuber Interaktif (PCV Project)

## Deskripsi Umum

Proyek ini mengimplementasikan aplikasi Virtual YouTuber (VTuber) berbasis real-time yang memadukan tracking wajah dan pose menggunakan MediaPipe dengan rendering model Live2D melalui Pygame dan OpenGL. Proyek ini dibagi menjadi dua tugas utama yang berfokus pada kontrol visual dan pemicu aksi otomatis.

## Struktur Proyek

### File: tugas1.py

**Fokus Utama: Kontrol Filter Video Real-time**

File ini berfokus pada integrasi efek visual (image filtering) pada video input dari webcam dan kontrolnya menggunakan keyboard.

**Fungsionalitas Kunci:**

1. **Implementasi Filter Gambar**
   
   Menerapkan tiga jenis filter utama pada frame video:
   - Average Blur dengan toggle kernel 5x5 dan 9x9
   - Gaussian Blur diterapkan melalui konvolusi dengan cv2.filter2D()
   - Sharpening menggunakan kernel kustom K_sharpen

2. **Kontrol Keyboard**
   
   Memungkinkan pengguna untuk beralih mode filter secara real-time:
   - Tombol '0': Mode Normal (tanpa filter)
   - Tombol '1': Average Blur
   - Tombol '2': Gaussian Blur
   - Tombol '3': Sharpening

3. **Tracking Dasar**
   
   Menangani tracking wajah dan pose tubuh untuk menggerakkan model Live2D.

### File: tugas2.py

**Fokus Utama: Pemicu Aksi Otomatis Berdasarkan Deteksi Warna**

File ini berfokus pada penggunaan teknik image processing lanjutan untuk deteksi objek berwarna dan menggunakannya sebagai pemicu otomatis (trigger) untuk aksi model Live2D.

**Fungsionalitas Kunci:**

1. **Deteksi Warna**
   
   Menerapkan proses deteksi objek berwarna Biru dan Hijau melalui langkah-langkah:
   - Konversi HSV dan Thresholding Warna
   - Pembersihan Mask menggunakan operasi Morfologi (Opening dan Closing)
   - Deteksi Kontur menggunakan cv2.findContours untuk memvalidasi keberadaan objek

2. **Aksi Pemicu**
   
   Memicu aksi berbeda pada model Live2D berdasarkan warna yang terdeteksi (Hijau memiliki prioritas lebih tinggi):
   - Jika objek Biru terdeteksi: memicu gerakan Lengan (ParamArmLA01, ParamArmRA01)
   - Jika objek Hijau terdeteksi: memicu ekspresi Buka Mulut (ParamA)

3. **Integrasi Otomatis**
   
   Mengintegrasikan pemicu deteksi warna ini secara langsung ke dalam loop aplikasi, sehingga model bereaksi secara otomatis terhadap lingkungan nyata.

## Teknologi yang Digunakan

- **OpenCV**: Pengambilan frame webcam dan image processing
- **MediaPipe**: Framework machine learning untuk deteksi landmark wajah, pose, dan tangan
- **Live2D SDK v3**: Rendering dan kontrol model avatar 2D
- **Pygame**: Manajemen window dan event handling
- **PyOpenGL**: Rendering grafis dan texture background
- **NumPy**: Operasi matematis dan array processing

## Fitur Tracking

### Tracking Kepala
- Rotasi Yaw (kiri-kanan)
- Rotasi Pitch (atas-bawah)
- Rotasi Roll (kemiringan)

### Tracking Mata
- Deteksi kedipan menggunakan Eye Aspect Ratio
- Tracking posisi bola mata dan arah pandangan

### Tracking Mulut
- Deteksi apertura mulut berdasarkan jarak bibir

### Tracking Tubuh
- Body Tilt (kemiringan horizontal)
- Body Roll (rotasi tubuh)
- Body Pitch (gerakan maju-mundur)

### Tracking Lengan
- Parameter shoulder (gerakan naik-turun)
- Parameter elbow (gerakan kiri-kanan)

### Tracking Kaki
- Deteksi posisi lutut untuk animasi kaki

## Sistem Smoothing

Seluruh parameter tracking menggunakan exponential moving average untuk menghaluskan gerakan:

Formula: nilai_baru = alpha x nilai_raw + (1 - alpha) x nilai_lama

Nilai alpha per parameter:
- Parameter kepala: 0.4
- Parameter mata: 0.6
- Parameter bola mata: 0.2
- Parameter tubuh: 0.2-0.3
- Parameter lengan: 0.6
- Parameter kaki: 0.4

## Mode Operasi

### Mode Statis
Model tetap di posisi tengah dengan skala default. Hanya orientasi tubuh yang berubah tanpa pergerakan posisi 2D atau zoom.

### Mode Dinamis
Model dapat bergerak mengikuti pergerakan tubuh pengguna dengan fitur:
- Translasi horizontal dan vertikal
- Auto-zoom berdasarkan jarak dari kamera
- Offset vertikal adaptif sesuai skala

Toggle antara mode dilakukan dengan menekan tombol S.

## Cara Penggunaan

### Persyaratan Sistem
- Python 3.x
- Webcam aktif
- GPU untuk performa optimal

### Instalasi Dependencies
```bash
pip install opencv-python
pip install mediapipe
pip install pygame
pip install PyOpenGL
pip install numpy
```

### Menjalankan Aplikasi

**Tugas 1 (Filter Video):**
```bash
python tugas1.py
```

**Tugas 2 (Deteksi Warna):**
```bash
python tugas2.py
```

### Kontrol Keyboard

**Tugas 1:**
- 0: Mode Normal
- 1: Average Blur
- 2: Gaussian Blur
- 3: Sharpening
- S: Toggle Static/Dynamic Model
- ESC: Keluar aplikasi

**Tugas 2:**
- S: Toggle Static/Dynamic Model
- ESC: Keluar aplikasi
Fokus Utama: Pemicu Aksi Otomatis Berdasarkan Deteksi Warna.

Fungsionalitas Kunci:
1. Deteksi Warna: Menerapkan proses deteksi objek berwarna Biru dan Hijau melalui langkah-langkah:
   - Konversi HSV dan Thresholding Warna.
   - Pembersihan Mask menggunakan operasi Morfologi (Opening dan Closing).
   - Deteksi Kontur (cv2.findContours) untuk memvalidasi keberadaan objek.
2. Aksi Pemicu: Memicu aksi berbeda pada model Live2D berdasarkan warna yang terdeteksi (Hijau memiliki prioritas lebih tinggi):
   - Jika objek Biru terdeteksi, memicu gerakan Lengan (ParamArmLA01, ParamArmRA01).
   - Jika objek Hijau terdeteksi, memicu ekspresi Buka Mulut (ParamA).
3. Integrasi Otomatis: Mengintegrasikan pemicu deteksi warna ini secara langsung ke dalam loop aplikasi, sehingga model bereaksi secara otomatis terhadap lingkungan nyata


