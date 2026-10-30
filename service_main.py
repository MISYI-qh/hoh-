"""Layanan latar belakang Android (foreground service): HOH tetap tumbuh walau aplikasi ditutup dan layar mati.
Mengerjakan loop Tumbuh (jelajah, riset, latih diri, notifikasi) di proses terpisah; UI hanya membaca/menulis berkas bersama."""
import os
os.environ["HOH_ROLE"] = "service"
import server

if __name__ == "__main__":
    g = server.E.grower(thread=False)
    g.run_forever()
