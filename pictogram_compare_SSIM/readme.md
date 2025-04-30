# Medical Pictogram Vision Demo

A tiny proof‑of‑concept that **captures an image from a HikVision industrial camera, finds the printed package, detects its orientation, crops individual pictograms and validates them against reference PNGs using SSIM**.

```
main.py ─┐  orchestrates everything, shows live preview
│
├ camera_thread.py   handles connection & streaming from GigE/USB HikVision camera (MvCamera SDK)
└ functions.py       all CV helpers: contour geometry, cropping, SSIM matching, heat‑map visualisation
```

---
## 1. Installation

```bash
# 1️⃣ Python >=3.9 in a venv
python -m venv .venv && . .venv/bin/activate   # PowerShell: .venv\Scripts\Activate.ps1

# 2️⃣ Core deps
pip install opencv-python numpy scikit-image 

# 3️⃣ HikVision SDK (Windows)
#   – download and install "MVS" package from Hikrobot site
#   – add \Program Files\MVS\Development\MVImport to PYTHONPATH or copy to project
```

> **GPU acceleration** – install `opencv-python‑headless` + CUDA build if you want to run on GPU.

---
## 2. Folder layout

```
project/
│  main.py             demo entry‑point
│  camera_thread.py     <- live video thread via MvCamera SDK
│  functions.py         <- all computer‑vision utilities
│  correct_source/      <- reference pictograms (PNG)
│  captured_image.png   <- optional dumps during debugging
└─ requirements.txt     <- optional pin
```

---
## 3. Quick start

```bash
python main.py           # Uses the HikVision camera by default
```

`settings = {"use_camera": True}` in **main.py** – flip to `False` if you only want to test on a static picture:

```python
image_package_path = "TraumastemTafLight.png"
settings["use_camera"] = False
```

The window shows:
* polygon overlay of the detected package
* centre coordinates + absolute rotation
* bounding boxes for all 14 known pictograms

Console prints mapping to a **0‑200 × 0‑200 mm** coordinate system and SSIM score for every pictogram.

---
## 4. Key modules

### camera_thread.py
* Enumerates **GigE / USB** devices via `MvCameraControl_class` (HikRobot SDK).
* Configures exposure, gain, frame‑rate.
* Runs a background thread → `latest_frame` always contains newest `numpy.ndarray` (Mono8).
* Optional callback lets you inject custom processing.

### functions.py
| Area | Function | Purpose |
|------|----------|---------|
| **Geometry** | `get_biggest_polygon` | finds outer contour (package) & rotation |
| | `get_absolute_angle` | PCA‑based orientation |
| | `map_coordinates` | maps camera px → mm (0‑200 range) |
| **Pre‑processing** | `detect_edges_and_contours`, `resize_frame`, `convert_to_black_and_white`, `remove_white_borders` | cleaning helpers |
| **Cropping** | `crop_rotated_rectangle`, `crop_rectangle`, `crop_square` | extract ROIs / pictograms |
| **Red box logic** | `extract_red_box_and_find_contours` | small heuristic that decides if crop is upside‑down |
| **Quality check** | `match_ssim`, `compare_frame_image_ssim`, `ssim_heatmap` | similarity score + visual heat‑map |
| **Debug** | `draw_grid` | draws a grid overlay |

### main.py
* Captures single frame every second → detects package → prints mapped coordinates & angle.
* Press **q** in any OpenCV window or hit **Ctrl‑C** to exit.

---
## 5. Extending

* **Batch mode** – change `CameraThread.capture_single_frame()` to continuous grabbing + queue.
* **Different pictogram set** – swap PNGs in **correct_source/** and update `hash_map_pictograms` in `functions.py`.
* **Better matching** – replace SSIM with template‑matching, CNN classifier or anomaly‑detection model.

---
## 6. Troubleshooting

| Issue | Hint |
|-------|------|
| `No devices found.` | Check camera is on, same subnet, firewall off.
| OpenCV window freezes | Release with **q**; long operations block UI thread.
| `Nedostatek shod pro výpočet homografie.` | Increase ORB features or ensure template & camera frame overlap.

---
## 7. License
use freely, but **at your own risk** in production medical environments.
