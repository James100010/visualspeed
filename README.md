# visualspeed
This program was primarly coded by AI.  
My intent behind this program is to be able to measure/estimate the speed of cars passing on the street in front.

For best results: 
1) Ensure the camera is stable (place it on a ledge or stand)
2) Setup the camera as perpenicular as you can to the road, with a good field of view of at least 4 car lenghts
3) Use easy to see marking points and measure the distance on the road between them.  (in my case I counted the 1 meter curb stones) 

Plans for next updates: 
1) Add real time video feed processing 
2) Add export processed video function (export showing object information) 


# AI Generated Readme : 
Visual Speed Measuring / Car Speed Detector

A PyQt6 desktop app for measuring vehicle speed in any video clip. **Version 2024‑07‑30** rewrites the processing core for higher FPS and safer threading.

---

## Features

- Draw **start** and **finish** lines directly on the first frame.
- KCF‑based multi‑object tracking .
- Optional sub‑frame interpolation for high‑precision timestamps.
- Live preview while the background thread analyses each frame.
- Results table (direction · elapsed time [s] · speed [km/h]) ready for copy‑paste to spreadsheets.

---

## Requirements

Python ≥ 3.9 and the following pip packages:

```bash
pip install PyQt6 opencv-contrib-python numpy
```

GPU acceleration is **not** required.

---

## Quick Start

```bash
git clone <repo‑url>
cd speed2
python carspeed.py
```

### In‑app workflow

1. **Load Video** – choose an MP4/AVI/MOV.
2. **Set Start Line** – click two points.
3. **Set Finish Line** – click two points.
4. Enter the real‑world **distance (m)** between the lines.
5. Click **Process Video**.\
   Speeds appear in the table on the right.

You can pause/seek the video at any time, even while placing lines.

---

## Tuning Parameters

| Setting            | Typical range | What it does                                                   |
| ------------------ | ------------- | -------------------------------------------------------------- |
| Min Object Area    | 300 – 3000 px | Ignore blobs smaller than this.                                |
| Motion Threshold   | 1 – 5 %       | Fraction of foreground pixels that counts as "moving".         |
| Max Static Frames  | 10 – 60       | How long a tracker can stay almost still before being dropped. |
| Detection Interval | 5 – 30 frames | How often to look for new vehicles. Lower = heavier CPU.       |

---

## File Overview

| File          | Purpose                     |
| ------------- | --------------------------- |
| `carspeed.py` | Main GUI + processing logic |
| `README.md`   | This quick‑start guide      |
| `LICENSE`     | Apache‑2.0 legal text       |

---

## License

Distributed under the **Apache License 2.0**. See the `LICENSE` file for full terms.

