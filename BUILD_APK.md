# Build APK HOH
Sama seperti app Weets-mu: Python + NumPy lewat buildozer (p4a, bootstrap webview).

**Lewat GitHub (tanpa PC):** buat repo, upload seluruh isi folder ini (termasuk `.github`), buka Actions → apk → Run workflow,
unduh artifact `HOH-apk`. Workflow memakai action pihak ketiga `ArtemSBulgakov/buildozer-action` dan belum pernah dijalankan oleh pembuatnya.

**Lokal/Termux/Linux:** `buildozer android debug` (pakai `buildozer.spec` di folder ini).

Model bawaan: `model/hoh-starter.hh`. Ganti dengan model `.hh` milikmu sebelum build kalau mau.

## Latar belakang di Android (belum teruji)
`buildozer.spec` mendefinisikan layanan `Hoh:service_main.py:foreground:sticky` dan izin notifikasi/latar depan. Agar tidak dimatikan sistem: Pengaturan HP → Aplikasi → HOH →
Baterai → "Tidak dibatasi", atau tekan "Izinkan berjalan di latar belakang" di kartu Tumbuh. Android atau pabrikan HP tertentu tetap bisa mematikan layanan; kalau build gagal
karena layanan/izin, kirim log-nya.
