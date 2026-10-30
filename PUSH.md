# Push ke GitHub (dari Termux/PC yang punya internet)
Repo lokal sudah berisi satu commit. Buat repo kosong `hoh` di akun MISYI-qh (jika namamu memang `hoh-`, ganti URL-nya), lalu:

    git remote set-url origin https://github.com/MISYI-qh/hoh.git
    git push -u origin main

Setelah ter-push: tab Actions → apk → Run workflow → unduh artifact `HOH-apk`.
