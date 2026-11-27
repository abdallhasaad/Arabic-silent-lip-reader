# إعداد وتشغيل ArabicLipGeometryNet v2 على لابتوب Windows آخر

هذا الملف يجمع أوامر PowerShell لإعداد بيئة مستقلة على اللابتوب الثاني، والتحقق من النموذج، ثم اختبار كاميرا الجهاز. نفّذ الأوامر بالترتيب.

> **مهم:** لا تنقل مجلد `.venv` من اللابتوب الأول. سننشئ بيئة افتراضية جديدة على الجهاز الثاني. ولا نحتاج إلى نقل بيانات LRW-AR أو ملفات التدريب؛ نحتاج فقط إلى حزمة النموذج وملفات معالجة الكاميرا.

---

## 1. نقل ملفات المشروع

انسخ مجلد المشروع من اللابتوب الأول إلى اللابتوب الثاني، مثلًا إلى:

```text
C:\Users\<USERNAME>\Downloads\ArabicLipGeometryNet_v2_Windows
```

يُفترض أن يحتوي المجلد على الأقل على الملفات التالية:

```text
ArabicLipGeometryNet_v2_Windows/
├── requirements.txt
├── visper_vsr_base.pth
├── visper_inference.py
├── camera_evaluation.py
├── camera_preprocessor.py
├── camera_preview.py
└── assets/
    ├── visper/
    └── face_landmarker.task
```

انقل ملفات الكاميرا من مجلد المشروع الحالي. يحتاج أمر التشغيل أيضًا إلى `visper_inference.py` و`camera_evaluation.py` وملف الأوزان `visper_vsr_base.pth` ومجلد المصدر `assets/visper`. يحتفظ الأرشيف `ArabicVSR_Camera_Preprocessing_Stage.zip` بنسخة المعاينة القديمة.

افتح PowerShell داخل مجلد المشروع. من File Explorer افتح المجلد، واكتب `powershell` في شريط العنوان، ثم اضغط Enter.

تحقق من المسار الحالي:

```powershell
Get-Location
```

---

## 2. فحص Python

اعرض إصدارات Python التي يتعرف عليها Python Launcher:

```powershell
py --list
```

تحقق من Python 3.13:

```powershell
py -3.13 --version
```

تحقق من أن Python يعمل بنظام 64-bit:

```powershell
py -3.13 -c "import sys, struct; print(sys.version); print('Architecture:', struct.calcsize('P') * 8, 'bit')"
```

استخدم Python 3.13.x بنسخة 64-bit مع مكوّن Tcl/Tk لنافذة الكاميرا. بعد تثبيت Python افتح PowerShell من جديد.

---

## 3. إنشاء البيئة الافتراضية وتفعيلها

إذا ظهر الخطأ `did not find executable at` ومعه مسار Python على
الجهاز الأول، فالبيئة `.venv` المنقولة مرتبطة بتثبيت Python هناك.
تحقّق أولًا من أن `py -3.13 --version` يعمل على الجهاز الثاني.
إذا كان `python --version` يعرض Python 3.13، يمكنك استخدام
`python -m venv .venv` بدلًا من `py -3.13 -m venv .venv`.
تتوفر مثبتات Python من [صفحة تنزيل Python لويندوز](https://www.python.org/downloads/windows/).

من مجلد المشروع الذي يحتوي على `camera_preview.py`، احفظ البيئة
المنقولة باسم احتياطي وأنشئ بيئة محلية:

```powershell
if (Test-Path -LiteralPath .\.venv) {
    Rename-Item -LiteralPath .\.venv -NewName ('.venv-copied-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) -ErrorAction Stop
}
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe camera_preview.py
```

هذه الأوامر تشغّل Python من البيئة مباشرة. بعد نجاح التشغيل يمكنك
تجاوز خطوات تفعيل البيئة التالية.

نفّذ من داخل مجلد المشروع:

```powershell
py -3.13 -m venv .venv
```

اسمح بتشغيل سكربت التفعيل في نافذة PowerShell الحالية فقط:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

فعّل البيئة الافتراضية:

```powershell
.\.venv\Scripts\Activate.ps1
```

يجب أن يظهر `(.venv)` في بداية سطر PowerShell. تحقق من المفسّر المستخدم:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

حدّث pip:

```powershell
python -m pip install --upgrade pip
```

---

## 4. تثبيت متطلبات النموذج وMediaPipe

ثبّت متطلبات النموذج والكاميرا الموجودة في الحزمة.

```powershell
python -m pip install -r requirements.txt
```

### تثبيت الإصدارات نفسها التي استُخدمت في اختبار اللابتوب الأول

لجعل المقارنة بين الجهازين أقرب ما يمكن، ثبّت الإصدارات التي ظهرت في سجل اللابتوب الأول:

```powershell
python -m pip install "numpy==2.5.5" "torch==2.15.0" "mediapipe==1.2.3" "opencv-contrib-python==5.1.0.95"
```

تحقق من عدم وجود تعارضات واضحة في التبعيات:

```powershell
python -m pip check
```

---

## 5. التحقق من المكتبات وواجهات MediaPipe

نفّذ الأمر التالي كاملًا:

```powershell
python -c "import cv2, mediapipe as mp, numpy as np, torch; from mediapipe.tasks import python as mp_python; from mediapipe.tasks.python import vision as mp_vision; print('MediaPipe:', mp.__version__); print('OpenCV:', cv2.__version__); print('NumPy:', np.__version__); print('PyTorch:', torch.__version__); print('FaceLandmarker API: OK')"
```

الإصدارات المتوقعة المطابقة للابتوب الأول:

```text
MediaPipe: 1.2.3
OpenCV: 5.1.0
NumPy: 2.5.5
PyTorch: 2.15.0
FaceLandmarker API: OK
```

قد تظهر رسالة تفيد بأن Matplotlib يبني Font Cache لأول مرة؛ هذه ليست بالضرورة مشكلة.

---

## 6. التأكد من ملف Face Landmarker

أنشئ مجلد `assets` إن لم يكن موجودًا:

```powershell
New-Item -ItemType Directory -Force assets
```

إذا لم يكن الملف `assets\face_landmarker.task` موجودًا، نزّله:

```powershell
if (-not (Test-Path ".\assets\face_landmarker.task")) {
    Invoke-WebRequest `
      -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" `
      -OutFile ".\assets\face_landmarker.task"
}
```

تحقق من وجوده وحجمه:

```powershell
Get-Item .\assets\face_landmarker.task | Format-List Name, Length, FullName
```

يجب أن يكون حجمه عدة ميغابايت. إذا كان الملف موجودًا بالفعل وصحيحًا، لا حاجة إلى تنزيله مرة أخرى.

---

## 7. اختبار تحميل النموذج

نفّذ:

```powershell
python camera_preview.py --check-visper
```

النتيجة المتوقعة تتضمن:

```text
ViSpeR Arabic model passed the inference check on cpu.
```

هذا الاختبار يؤكد تحميل النموذج وتنفيذ الاستدلال على مدخل اصطناعي. **لا يقيس دقة قراءة الشفاه الحقيقية**.

لا تنتقل إلى اختبار الكاميرا إذا ظهر خطأ في تحميل النموذج أو الملفات.

---

## 8. السماح لتطبيقات سطح المكتب باستخدام الكاميرا

افتح إعدادات خصوصية الكاميرا:

```powershell
Start-Process "ms-settings:privacy-webcam"
```

تأكد من تفعيل الخيارات المتاحة التالية:

- Camera access
- Let apps access your camera
- Let desktop apps access your camera

---

## 9. اختبار كاميرا اللابتوب بالدقة الافتراضية

أغلق أي برنامج آخر يستخدم الكاميرا، ثم نفّذ:

```powershell
python camera_preview.py
```

افتح نافذة قارئ الشفاه وانظر باتجاه الكاميرا تحت إضاءة ثابتة. اضغط «ابدأ» أو `Space`، وانتظر «انطق الآن»، ثم انطق كلمة أو عبارة. استخدم «قياس الدقة» لمقارنة النص المقروء بنص معلوم. اضغط `Q` أو `Esc` للخروج.

إذا كانت الكاميرا المطلوبة تستخدم رقمًا آخر، نفّذ `python camera_preview.py --camera 1`.

---

## 10. اختبار الكاميرا بدقة 480×360

بعد إغلاق الاختبار السابق، نفّذ:

```powershell
python camera_preview.py --width 480 --height 360
```

جرّب الكلمات نفسها وسجّل عدد المحاولات التي تعرض نتيجة ورسائل نقص الإطارات إن ظهرت. اضغط `Q` أو `Esc` للإغلاق.

---

## 11. جمع مواصفات اللابتوب الثاني

### المعالج

```powershell
Get-CimInstance Win32_Processor |
Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
```

### الشركة المصنّعة والطراز والذاكرة

```powershell
Get-CimInstance Win32_ComputerSystem |
Select-Object Manufacturer, Model, TotalPhysicalMemory
```

### بطاقة الرسوميات

```powershell
Get-CimInstance Win32_VideoController |
Select-Object Name, AdapterRAM
```

### إصدار Windows

```powershell
winver
```

---

## 12. تقرير النتائج المطلوب إرساله

بعد إكمال الاختبارات، أرسل مخرجات الأوامر أو انسخ القيم في الجدول التالي:

| الاختبار | النتيجة على اللابتوب الثاني |
|---|---|
| Python version / Architecture | |
| PyTorch | |
| MediaPipe | |
| OpenCV | |
| `python -m pip check` | |
| `camera_preview.py --check-visper` | |
| معدل خطأ الكلمات بالدقة الافتراضية | |
| معدل خطأ الكلمات عند 480×360 | |
| عدد المحاولات التي أعادت نتيجة | |
| رسائل نقص الإطارات أو تعذّر تتبّع الشفاه | |
| CPU model | |
| RAM | |
| GPU | |

---

## ملاحظات مهمة

1. لا تنقل مجلد `.venv` بين الجهازين؛ أنشئه محليًا على كل جهاز.
2. استخدم ملف `visper_vsr_base.pth` نفسه عند مقارنة الجهازين.
3. لا نحتاج إلى تنزيل بيانات LRW-AR إلى اللابتوب الثاني.
4. استخدم «قياس الدقة» داخل نافذة الكاميرا لتقييم قراءة الكلمات والعبارات المنطوقة.
5. يحتفظ التطبيق بإطارات الكاميرا والإحداثيات في الذاكرة أثناء الاستخدام.
6. أجرِ الاختبارين على الجهازين بإضاءة ودقة كاميرا متقاربتين، وأغلق أي تطبيق آخر يستخدم الكاميرا.
