# Driver Drowsiness Detection — Data Slayer 3.0

Notebook computer vision untuk mengklasifikasikan kondisi **drowsiness** dan **non-drowsiness** dari video pengemudi.

## Pendekatan

- Ekstraksi frame video dengan OpenCV.
- Parsing subjek, sudut kamera, dan perilaku.
- Pembentukan sequence 30 frame.
- Crop area wajah dan augmentasi citra.
- Transfer learning menggunakan EfficientNet-B0.
- Subject-based train/validation split untuk mengurangi kebocoran data.
- Class-weighted binary cross-entropy.
- Sequence-level inference melalui agregasi probabilitas frame.
- Evaluasi menggunakan Macro F1.

## File

File `Data Slayer 3.0` berisi notebook JSON tanpa ekstensi. Agar mudah dibuka di Jupyter atau Kaggle, unduh lalu ubah namanya menjadi `data-slayer-3.ipynb`.

## Instalasi

```bash
pip install torch torchvision timm pandas numpy scikit-learn opencv-python pillow tqdm jupyter
```

## Menjalankan

1. Siapkan dataset kompetisi dengan folder `train/`, `test/`, dan `sample_submission.csv`.
2. Sesuaikan `BASE_DIR` pada notebook.
3. Jalankan ekstraksi frame dan pembentukan sequence.
4. Latih EfficientNet-B0.
5. Pilih threshold berdasarkan Macro F1 validasi.
6. Jalankan inference dan buat submission.

## Catatan Reproducibility

Notebook menggunakan seed 42. Hasil dapat berubah karena versi library, perangkat GPU, serta pembagian subjek. Catat versi environment saat membandingkan eksperimen.
