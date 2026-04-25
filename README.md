# Kinetic Nebula Web

This project turns your hands into a live gesture controller with a transparent in-air visual effect over your camera.

## What it does
- Index + middle fingers: move mouse cursor.
- Index finger only: click mouse button.
- Open palm: scroll up/down by moving your palm vertically.
- Close both hands (two fists) or join both palms (prayer pose): volume immediately goes to 0%.
- The same close/join gesture also freezes the frequency wave.
- After that, keep both palms open to control volume by hand gap.
  - Palms closer together: lower volume.
  - Palms farther apart: higher volume.
  - The app adapts your widest palm gap as the 100% limit (shoulder-gap style max).
- Show pinky finger only: opens Spotify (`spotify:` URI by default).
- Both palms open: shows an animated multidimensional frequency bridge between both hands.
  - Frequency follows live song/audio using system loopback (or mic fallback).

## Visual style
- Live camera with transparent-style neon trail overlay.
- White hand landmark points (clean point-style detection view).
- Mirror-water style dynamic palm-to-palm frequency waveform.
- Optional microphone reactivity for stronger bridge motion.
  - Audio reactive means the bridge frequency and motion react to live audio in real time.
  - `--audio-source auto` tries system-output loopback first (song), then microphone fallback.
- In dark scenes, night-boost is applied for brighter interactive visuals.
- Window is pinned to the left, requested as topmost, and shown as a small fixed preview.

## Setup
1. Activate your virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run
```powershell
python main.py
```

## Recommended command
```powershell
python main.py --width 1920 --height 1080 --audio-reactive
```

## Useful options
```powershell
python main.py --camera-alpha 170 --trail-alpha 28
python main.py --camera-index 1
python main.py --no-mouse-control
python main.py --no-media-keys
python main.py --preview-width 420 --preview-height 236
python main.py --pinky-app-uri spotify:
python main.py --audio-reactive --audio-source system --audio-frequency-weight 0.8
python main.py --hands-join-threshold 0.12 --volume-min-gap 0.08 --volume-gap-gain 2.2
```

## Gesture map summary
- `Index + Middle`: mouse move mode.
- `Index only`: click mode.
- `Palm open`: scroll mode.
- `Both fists` or `joined palms`: mute to 0% and freeze wave.
- `Both palms open`: adjust volume by palm distance with adaptive max.
- `Pinky only`: open Spotify app URI.
- `Both open palms`: animated frequency bridge active.

## Notes
- If camera access is blocked, the app exits with a clear error.
- If `pyautogui` is not available, visual effects still run and mouse actions are disabled.
- If `assets/hand_landmarker.task` is missing, it is downloaded automatically on first run.
