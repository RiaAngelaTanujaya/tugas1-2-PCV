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
<img width="1280" height="720" alt="Saat Berwarna Biru" src="https://github.com/user-attachments/assets/35b89ffc-7413-47de-b02a-59cdfcb5189b" />

