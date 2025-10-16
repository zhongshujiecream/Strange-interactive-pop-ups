"""随机细胞生成器 — Random Cell Generator

运行: python random_cat_generator.py

生成逻辑:
用户在输入框输入一个数字（种子），点击 Generate，程序会用该种子生成一些像素风的“神奇细胞”并在画布上显示。
细胞包含细胞膜、细胞质、细胞核与若干细胞器与纤毛，颜色与形态受种子控制。
支持保存为 PNG（依赖 Pillow，可选）。
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
import math
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


CANVAS_PIXELS = 32  # pixel-art grid size (32x32)
SCALE = 12  # visual scale for canvas


def seeded_random(seed):
    rnd = random.Random(seed)
    return rnd


def generate_cat_grid(rnd, size=CANVAS_PIXELS):
    """Generate a more varied pixel cat with multiple style variants.

    Styles change the head shape, ear size, eye shape, body pose and tail length
    so different seeds produce visually distinct cat silhouettes.
    """
    grid = [[0 for _ in range(size)] for _ in range(size)]
    cx = size // 2

    def setp(x, y, v):
        if 0 <= x < size and 0 <= y < size:
            grid[y][x] = v

    def mset(x, y, v):
        setp(x, y, v)
        setp(size-1-x, y, v)

    # color palette choices
    fur_index = rnd.randint(0, 5)
    fur_colors = [
        (160, 160, 200), (200, 160, 120), (120, 90, 60),
        (220, 220, 220), (80, 80, 120), (180, 140, 180),
    ]
    fur_rgb = fur_colors[fur_index % len(fur_colors)]
    accent_rgb = tuple(min(255, c+50) for c in fur_rgb)
    stripe_rgb = tuple(max(0, c-40) for c in fur_rgb)

    # pick a high-level style
    style = rnd.choice(['round', 'chubby', 'siamese', 'tall', 'sitting'])

    # head/body placement parameters
    head_top = max(1, size//8)
    if style == 'round':
        head_h = size * 10 // 32
        head_w = size * 10 // 32
    elif style == 'chubby':
        head_h = size * 11 // 32
        head_w = size * 12 // 32
    elif style == 'siamese':
        head_h = size * 11 // 32
        head_w = size * 8 // 32
    elif style == 'tall':
        head_h = size * 12 // 32
        head_w = size * 9 // 32
    else:  # sitting
        head_h = size * 9 // 32
        head_w = size * 9 // 32

    head_left = cx - head_w//2
    head_right = cx + head_w//2
    head_bottom = head_top + head_h

    # draw head (ellipse-ish via mask)
    rx = head_w/2
    ry = head_h/2
    for y in range(head_top, head_bottom):
        for x in range(head_left, head_right+1):
            dx = x - cx
            dy = y - (head_top + ry)
            # normalized ellipse metric
            if (dx*dx)/(rx*rx + 1e-6) + (dy*dy)/(ry*ry + 1e-6) <= 1.0:
                setp(x, y, 1)

    # ears vary by style
    ear_h = max(2, size//12)
    if style == 'siamese':
        ear_h += 1
    for i in range(ear_h):
        # left
        ex = head_left + i
        ey = head_top - i - 1
        setp(ex, ey, 1)
        # right
        exr = head_right - i
        setp(exr, ey, 1)

    # eyes: shape changes
    eye_y = head_top + int(head_h*0.35)
    eye_dx = max(1, head_w//4)
    left_eye_x = cx - eye_dx
    right_eye_x = cx + eye_dx
    if style == 'siamese':
        # almond eyes (wider horizontally)
        for dx in (-1, 0, 1):
            setp(left_eye_x+dx, eye_y, 3)
            setp(right_eye_x+dx, eye_y, 3)
        setp(left_eye_x, eye_y, 5)
        setp(right_eye_x, eye_y, 5)
    else:
        # round eyes
        for dy in (0,1):
            setp(left_eye_x, eye_y+dy, 3)
            setp(right_eye_x, eye_y+dy, 3)
        setp(eye_y and left_eye_x, eye_y, 5)  # safe noop if out of bounds
        setp(right_eye_x, eye_y, 5)

    # nose and mouth
    nose_y = eye_y + 2
    setp(cx, nose_y, 4)
    if 0 <= nose_y+1 < size:
        setp(cx-1, nose_y+1, 4)
        setp(cx+1, nose_y+1, 4)

    # whiskers
    for off in (-1, 0, 1):
        y = nose_y + off
        if 0 <= y < size:
            for dx in range(2, 6):
                setp(cx-dx, y, 6)
                setp(cx+dx, y, 6)

    # body and pose
    if style == 'sitting':
        body_top = head_bottom
        body_left = cx - head_w
        body_right = cx + head_w
        body_bottom = size - 3
        for y in range(body_top, body_bottom):
            for x in range(body_left, body_right+1):
                setp(x, y, 1)
        # legs: two small blocks
        leg_y = body_bottom - 1
        for lx in (cx-2, cx+1):
            for x in range(lx, lx+2):
                setp(x, leg_y, 1)
    else:
        # standing/normal body
        body_top = head_bottom
        body_left = cx - head_w - 1
        body_right = cx + head_w + 1
        body_bottom = size - 3
        for y in range(body_top, body_bottom):
            for x in range(body_left, body_right+1):
                setp(x, y, 1)

    # belly accent
    belly_w = max(2, head_w//3)
    belly_left = cx - belly_w//2
    belly_right = cx + belly_w//2
    for y in range(body_top + 1, min(body_top + 1 + belly_w, size)):
        for x in range(belly_left, belly_right+1):
            setp(x, y, 2)

    # stripes/spots depending on random
    if rnd.random() < 0.6:
        for s in range(rnd.randint(2, 5)):
            sy = rnd.randint(body_top, min(body_top+6, size-2))
            sx = rnd.randint(1, head_w//2)
            for dx in range(1, rnd.randint(1, 4)):
                setp(cx-sx-dx, sy, 7)
                setp(cx+sx+dx, sy, 7)
    else:
        # calico patches
        for p in range(rnd.randint(1, 3)):
            px = rnd.randint(body_left+1, body_right-1)
            py = rnd.randint(body_top+1, body_bottom-2)
            setp(px, py, 7)
            setp(px+1, py, 7)

    # tail length variation
    tail_len = 3 + (0 if style == 'sitting' else rnd.randint(1, 4))
    tx = body_right
    ty = body_bottom - 2
    for i in range(tail_len):
        setp(tx + i, ty - (i//2), 1)

    # small decorative paw or spot
    if rnd.random() < 0.3:
        setp(body_bottom-2, cx- (head_w//2), 7)

    color_map = {
        0: (255, 255, 255),
        1: fur_rgb,
        2: accent_rgb,
        3: (240, 240, 240),
        4: (220, 120, 140),
        5: (10, 10, 10),
        6: (200, 200, 200),
        7: stripe_rgb,
    }

    return grid, color_map


def draw_grid_on_canvas(canvas, grid, color_map, scale=SCALE):
    canvas.delete("all")
    size = len(grid)
    canvas.config(width=size*scale, height=size*scale)
    for y in range(size):
        for x in range(size):
            v = grid[y][x]
            if v == 0:
                continue
            rgb = color_map.get(v, (0,0,0))
            hexc = '#%02x%02x%02x' % rgb
            x0 = x*scale
            y0 = y*scale
            canvas.create_rectangle(x0, y0, x0+scale, y0+scale, outline=hexc, fill=hexc)


def save_grid_as_png(grid, color_map, path, scale=1):
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow not available. Install pillow to save images: pip install pillow")
    size = len(grid)
    img = Image.new('RGBA', (size, size), (255,255,255,0))
    draw = ImageDraw.Draw(img)
    for y in range(size):
        for x in range(size):
            v = grid[y][x]
            if v == 0:
                continue
            rgb = color_map.get(v, (0,0,0))
            draw.point((x, y), fill=rgb+(255,))
    if scale != 1:
        img = img.resize((size*scale, size*scale), resample=Image.NEAREST)
    img.save(path)


class PixelCatApp:
    def __init__(self, root):
        self.root = root
        root.title('Random Cat Generator')

        frm = ttk.Frame(root, padding=12)
        frm.grid(row=0, column=0, sticky='nsew')

        self.seed_var = tk.StringVar()
        seed_label = ttk.Label(frm, text='Enter numeric seed:')
        seed_label.grid(row=0, column=0, sticky='w')
        seed_entry = ttk.Entry(frm, textvariable=self.seed_var, width=20)
        seed_entry.grid(row=0, column=1, sticky='w')
        seed_entry.bind('<Return>', lambda e: self.generate())

        gen_btn = ttk.Button(frm, text='Generate', command=self.generate)
        gen_btn.grid(row=0, column=2, padx=6)

        save_btn = ttk.Button(frm, text='Save', command=self.save_image)
        save_btn.grid(row=0, column=3)

        self.canvas = tk.Canvas(frm, width=CANVAS_PIXELS*SCALE, height=CANVAS_PIXELS*SCALE, bg='white')
        self.canvas.grid(row=1, column=0, columnspan=4, pady=8)

        note = ttk.Label(frm, text='Enter a number seed and click Generate. Each seed produces a different cat pattern!')
        note.grid(row=2, column=0, columnspan=4)

        # initial
        self.current_grid = None
        self.current_color_map = None

    def generate(self):
        s = self.seed_var.get().strip()
        if s == '':
            messagebox.showinfo('Notice', 'Please enter a numeric seed (e.g., 42)')
            return
        try:
            # allow big ints
            seed = int(s)
        except Exception:
            # fall back to hash of string
            seed = abs(hash(s)) % (2**32)

        rnd = seeded_random(seed)
        grid, cmap = generate_cat_grid(rnd)
        self.current_grid = grid
        self.current_color_map = cmap
        draw_grid_on_canvas(self.canvas, grid, cmap)

    def save_image(self):
        if self.current_grid is None:
            messagebox.showwarning('Warning', 'Please generate a cell before saving.')
            return
        initial = 'pixel_cell.png'
        path = filedialog.asksaveasfilename(defaultextension='.png', initialfile=initial, filetypes=[('PNG','*.png')])
        if not path:
            return
        try:
            save_grid_as_png(self.current_grid, self.current_color_map, path, scale=8)
            messagebox.showinfo('Saved', f'Saved to: {path}')
        except Exception as e:
            messagebox.showerror('Error', str(e))


def main():
    try:
        root = tk.Tk()
    except Exception:
        import traceback
        with open('run_error.txt', 'w', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        print('Failed to create Tk root. Traceback written to run_error.txt')
        raise

    try:
        app = PixelCatApp(root)
        root.mainloop()
    except Exception:
        import traceback
        with open('run_error.txt', 'w', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        print('Exception occurred during app run. Traceback written to run_error.txt')
        raise


if __name__ == '__main__':
    main()
