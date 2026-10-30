[app]
title = HOH
package.name = hoh
package.domain = app.hoh
source.dir = .
source.include_exts = py,html,json,svg,png,hh,jsonl,wav
source.exclude_dirs = tests,bin,.buildozer,hoh_home
version = 1.2.0
requirements = python3,numpy,pyjnius,android
icon.filename = %(source.dir)s/icon.png
presplash.filename = %(source.dir)s/presplash.png
orientation = portrait
android.permissions = INTERNET,POST_NOTIFICATIONS,FOREGROUND_SERVICE,WAKE_LOCK,REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,VIBRATE,RECORD_AUDIO
android.api = 33
android.minapi = 24
services = Hoh:service_main.py:foreground:sticky
android.archs = arm64-v8a
android.presplash_color = #FFFFFF
p4a.bootstrap = webview
android.port = 5000

[buildozer]
log_level = 2
