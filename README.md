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
