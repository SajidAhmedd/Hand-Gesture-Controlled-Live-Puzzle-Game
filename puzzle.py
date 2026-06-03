import cv2
import numpy as np
import random
import time

class Tile:
    def __init__(self, tid, image, correct_pos, current_pos, tile_w, tile_h):
        self.tid = tid
        self.image = image
        self.correct_pos = correct_pos
        self.current_pos = current_pos
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.is_dragging = False
        self.drag_x = 0
        self.drag_y = 0

    def is_correct(self):
        return self.current_pos == self.correct_pos

    def get_screen_pos(self, offset_x, offset_y):
        if self.is_dragging:
            return (self.drag_x, self.drag_y)
        row, col = self.current_pos
        return (offset_x + col * self.tile_w, offset_y + row * self.tile_h)

class PuzzleBoard:
    def __init__(self, rows=3, cols=3):
        self.rows = rows
        self.cols = cols
        self.tiles = []
        self.tile_w = 0
        self.tile_h = 0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging_tile = None
        self.is_solved = False
        # Cooldown to prevent rapid pickup after drop
        self.last_drop_time = 0
        self.pickup_cooldown = 0.3  # 300ms cooldown after dropping

    def initialize_from_image(self, image, offset_x, offset_y):
        h, w = image.shape[:2]
        self.tile_w = w // self.cols
        self.tile_h = h // self.rows
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.tiles = []
        tid = 0
        for row in range(self.rows):
            for col in range(self.cols):
                y1, y2 = row * self.tile_h, (row + 1) * self.tile_h
                x1, x2 = col * self.tile_w, (col + 1) * self.tile_w
                tile_img = image[y1:y2, x1:x2].copy()
                tile = Tile(tid, tile_img, (row, col), (row, col), self.tile_w, self.tile_h)
                self.tiles.append(tile)
                tid += 1
        self.dragging_tile = None
        self.is_solved = False
        self.last_drop_time = 0

    def shuffle(self):
        positions = [t.current_pos for t in self.tiles]
        random.shuffle(positions)
        for tile, pos in zip(self.tiles, positions):
            tile.current_pos = pos
        if self.check_solved():
            p0, p1 = self.tiles[0].current_pos, self.tiles[1].current_pos
            self.tiles[0].current_pos, self.tiles[1].current_pos = p1, p0
        self.is_solved = False
        self.dragging_tile = None

    def get_tile_at_screen_pos(self, x, y):
        for tile in self.tiles:
            if tile.is_dragging:
                continue
            tx, ty = tile.get_screen_pos(self.offset_x, self.offset_y)
            if tx <= x < tx + self.tile_w and ty <= y < ty + self.tile_h:
                return tile
        return None

    def get_grid_pos_at_screen(self, x, y):
        gx = x - self.offset_x
        gy = y - self.offset_y
        col = gx // self.tile_w
        row = gy // self.tile_h
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return (row, col)

    def handle_input(self, cursor_pos, is_pinching):
        if cursor_pos is None:
            return

        cx, cy = cursor_pos
        current_time = time.time()

        if is_pinching:
            if self.dragging_tile:
                # Continue dragging
                self.dragging_tile.drag_x = cx - self.tile_w // 2
                self.dragging_tile.drag_y = cy - self.tile_h // 2
            else:
                # Check cooldown before picking up
                if current_time - self.last_drop_time >= self.pickup_cooldown:
                    # Try to pick up a tile
                    tile = self.get_tile_at_screen_pos(cx, cy)
                    if tile:
                        tile.is_dragging = True
                        tile.drag_x = cx - self.tile_w // 2
                        tile.drag_y = cy - self.tile_h // 2
                        self.dragging_tile = tile
        else:
            if self.dragging_tile:
                self.drop_tile()

    def drop_tile(self):
        if not self.dragging_tile:
            return

        tile = self.dragging_tile
        cx = tile.drag_x + self.tile_w // 2
        cy = tile.drag_y + self.tile_h // 2

        new_row, new_col = self.get_grid_pos_at_screen(cx, cy)
        old_pos = tile.current_pos
        new_pos = (new_row, new_col)

        # Find tile at new position and swap
        if new_pos != old_pos:
            for other in self.tiles:
                if other != tile and other.current_pos == new_pos:
                    other.current_pos = old_pos
                    break
            tile.current_pos = new_pos

        tile.is_dragging = False
        self.dragging_tile = None
        self.last_drop_time = time.time()  # Record drop time for cooldown
        self.is_solved = self.check_solved()

    def check_solved(self):
        return all(t.is_correct() for t in self.tiles)

    def draw(self, canvas):
        # Draw non-dragging tiles first
        for tile in self.tiles:
            if not tile.is_dragging:
                self.draw_tile(canvas, tile)

        # Draw dragging tile on top
        if self.dragging_tile:
            self.draw_tile(canvas, self.dragging_tile, highlight=True)

    def draw_tile(self, canvas, tile, highlight=False):
        x, y = tile.get_screen_pos(self.offset_x, self.offset_y)
        x, y = int(x), int(y)

        # Clamp to canvas bounds
        h, w = canvas.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + self.tile_w)
        y2 = min(h, y + self.tile_h)

        if x2 > x1 and y2 > y1:
            tx1 = x1 - x
            ty1 = y1 - y
            tx2 = tx1 + (x2 - x1)
            ty2 = ty1 + (y2 - y1)
            canvas[y1:y2, x1:x2] = tile.image[ty1:ty2, tx1:tx2]

        # Border
        if highlight:
            cv2.rectangle(canvas, (x, y), (x+self.tile_w-1, y+self.tile_h-1), (0, 255, 0), 3)
        else:
            cv2.rectangle(canvas, (x, y), (x+self.tile_w-1, y+self.tile_h-1), (80, 80, 80), 1)
