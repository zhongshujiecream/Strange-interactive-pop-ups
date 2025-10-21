Random Pixel Cat Generator

This is a small pixel-art generator written in Python using Tkinter. Enter a numeric seed in the input box and click "Generate" — the program will produce a pixel-style cat based on the seed and display it in the window.

Files
- `random_cat_generator.py` — main program (GUI and generator).

Dependencies
- Python 3.x
- Tkinter (usually included with Python)
- Pillow (optional, required for saving PNGs):
	pip install pillow

Run
Open a terminal (Windows PowerShell), change to this project directory and run:

```powershell
python random_cat_generator.py
```

Usage
- Enter an integer seed (for example: 42) and click "Generate".
- Click "Save" to export the generated cat as a PNG (requires Pillow).

Notes
- The same numeric seed will produce the same cat image (deterministic).
- Feel free to edit the generator parameters (colors, grid size) to create different styles.

License
- Free to use and modify. Contributions welcome.

# 3D Dream Generator

This project is a 3D dream visualizer built with Python 3 and pygame.

## Features
- Dynamic generation of 3D cubes, floating buildings, spheres, dream tunnels, and more
- Mouse interaction: affect rotation, floating, repulsion, etc.
- Color gradients, semi-transparent trails, twinkling stars for a dreamy effect
- Windows supported

## Requirements
- Python 3.12.x (recommended; Python 3.14 is not supported)
- pygame

## Install dependencies
```powershell
python -m pip install pygame
```

## How to run
Open a terminal in this project directory and run:
```powershell
python dream.py
```
Or, if you use a specific Python 3.12 path:
```powershell
C:\Users\zsj\AppData\Local\Programs\Python\Python312\python.exe dream.py
```

## Files
- `dream.py`: Main program, 3D Dream Generator

## FAQ
- If pygame installation fails, make sure you are using Python 3.12 or 3.11.
- If the `python` command is not recognized, use the full path to your Python executable.

## Preview
When running, a window will pop up showing a dynamic 3D dream world.

---
If you have any questions, feel free to ask!
