# HOH — aplikasi + web

HOH adalah anak dari perusahaan Misyi (buatan Faqih Musyaffa): filsafat dari seluruh sudut pandang makhluk hidup, mesin, dan benda mati, bernada puitis.

Jalankan: `python main.py` lalu buka http://localhost:8765 (web dan aplikasi memakai kode yang sama).
Android: `buildozer android debug` (server di port 5000, dibuka WebView). Panduan build lewat GitHub ada di `BUILD_APK.md`.

## HOH hanya menganalisis — tidak membuat apa pun
- Tidak ada pembuat gambar, berkas, maupun kode; tidak ada penjalan kode.
- Bisa: obrolan, **analisis berkas apa saja** (ketuk + lalu Unggah berkas: pdf/docx/xlsx/teks/gambar/audio dibaca isinya; arsip, program, dan biner dideskripsikan: tipe, ukuran, SHA-256, entropi, isi arsip, string terbaca — tidak pernah dijalankan), cari web, buka halaman, hitung, waktu, dan belajar dari ajaranmu.

## Model `.hh`
`model/hoh-starter.hh` (1,6 juta parameter) dipasang otomatis sebagai model **Bawaan**. Halaman **Model**:
- daftar model: Bawaan, Hasil latih, Impor — pilih **Pakai** untuk berpindah kapan saja; bawaan tidak bisa terhapus;
- **Ekspor** per model ke berkas `.hh`; **Impor** hanya menerima `.hh` buatan HOH (magic, checksum, tanda tangan, bentuk tensor, nilai berhingga, tanpa pickle);
- model impor tidak otomatis dipakai: setelah impor kamu ditanya mau dipakai atau tetap model yang sekarang.
Untuk memakai modelmu sendiri: ekspor dari HOH lain, atau ganti `model/hoh-starter.hh` sebelum build.

## Latih
Halaman Latih: tanya-jawab, teks, atau tautan, dengan preset Cepat/Seimbang/Dalam. Pelatihan selalu melanjutkan model aktif dan hasilnya jadi slot "Hasil latih".


## Suara
HOH bisa membaca jawabannya dengan suara (tombol speaker di tiap jawaban, atau otomatis di Pengaturan). Mesin suara (`hoh/voice.py`, NumPy saja) mengubah ejaan
Indonesia menjadi fonem lalu menyintesis formant. Halaman Latih → **Suara**: unggah/rekam suara (30–90 dtk) dan HOH mengukur pitch, panjang saluran suara,
warna spektrum, napas, dan kecepatan; hasilnya 32 angka yang disimpan DI DALAM model (`voice/profile`), jadi ikut berpindah bersama berkas `.hh`.
Ini bukan kloning suara dan terdengar seperti robot; kualitas pengucapan belum bisa diverifikasi oleh pembuatnya (tidak bisa mendengar).

## Tumbuh sendiri (latar belakang)
`hoh/grow.py`: saat tidak ada yang mengobrol, HOH menjelajah web (maks 5 pencarian per siklus jika tidak disuruh), membaca dengan sopan (robots.txt, jeda,
kuota harian 60 halaman), menyimpan catatan, lalu melatih diri singkat. Hasil latih mandiri hanya dipakai jika tidak lebih buruk pada soal jangkar; model lama tetap ada
sebagai slot. Riset bebas (hingga 25 pencarian) dimulai dari kartu Tumbuh atau saat model memilih menulis `<riset>…</riset>`; hasilnya masuk kotak masuk dan dipakai menjawab saat ada yang bertanya lagi.
Jempol suka dilatihkan; jempol tidak suka tidak pernah dilatihkan dan pertanyaannya menjadi target riset (teks pertanyaan itu dikirim ke mesin pencari).
Matikan semuanya dari kartu Tumbuh (saklar) kapan saja. Di Android loop ini berjalan di layanan latar depan (`service_main.py`); belum pernah diuji di perangkat.


## Protokol aksi: model yang memutuskan
Model menulis langkahnya sendiri sebagai tag di aliran teks: `<think>` (berpikir), `<cek>` (memeriksa), `<cari>q</cari>` (cari di web), `<riset>t</riset>` (riset bebas di latar belakang),
`<ingat>t</ingat>` (ambil catatan riset). UI membaca fase dari tag yang SEDANG ditulis model (bukan pewaktu); runtime hanya menjalankan alat yang dipilih model lewat tag.
Router niat berbasis kata kunci di sisi klien dihapus (kecuali mode web yang dipilih pengguna). Model dilatih satu giliran (tanya→jawab), jadi hanya pesan terakhir yang dikirim.
Jujur soal mutu: model 1,6 juta parameter ini menghafal pola latihan; tag dipakai dengan benar pada pola yang dilatih, tetapi isi tag (topik pencarian, hasil hitungan, "pemeriksaan") sering salah pada soal baru.

## Data latih bawaan
`model/seed.jsonl`: 5.000 contoh bahasa Indonesia, sebagian besar dibangkitkan dengan kode dari daftar dan templat tulisan tangan (aritmetika, angka-kata, hari/bulan, perbandingan,
bahasa, ibu kota provinsi, identitas, filsafat/puisi, pemakaian alat). Contoh bukan 5.000 teks independen: ±3.100 jawaban unik.
