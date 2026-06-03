"""
Tile class for the Live Puzzle game.
Each tile represents a piece of the puzzle.
"""

import cv2
import numpy as np


class Tile:
    def __init__(self, tile_id: int, image: np.ndarray, correct_row: int, correct_col: int,
                 current_row: int, current_col: int, tile_width: int, tile_height: int):
        """
        Initialize a puzzle tile.

        Args:
            tile_id: Unique identifier for the tile
            image: The image data for this tile
            correct_row: The correct row position (for win detection)
            correct_col: The correct column position (for win detection)
            current_row: Current row position on the grid
            current_col: Current column position on the grid
            tile_width: Width of the tile in pixels
            tile_height: Height of the tile in pixels
        """
        self.tile_id = tile_id
        self.image = image
        self.correct_row = correct_row
        self.correct_col = correct_col
        self.current_row = current_row
        self.current_col = current_col
        self.tile_width = tile_width
        self.tile_height = tile_height

        # Pixel position (for dragging)
        self.x = current_col * tile_width
        self.y = current_row * tile_height

        # Drag state
        self.is_selected = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0

    def update_grid_position(self, row: int, col: int):
        """Update the tile's grid position."""
        self.current_row = row
        self.current_col = col
        self.x = col * self.tile_width
        self.y = row * self.tile_height

    def snap_to_grid(self, grid_cols: int, grid_rows: int):
        """Snap the tile to the nearest grid position."""
        # Calculate nearest grid position
        new_col = round(self.x / self.tile_width)
        new_row = round(self.y / self.tile_height)

        # Clamp to valid grid bounds
        new_col = max(0, min(new_col, grid_cols - 1))
        new_row = max(0, min(new_row, grid_rows - 1))

        return new_row, new_col

    def is_in_correct_position(self) -> bool:
        """Check if the tile is in its correct position."""
        return self.current_row == self.correct_row and self.current_col == self.correct_col

    def contains_point(self, px: int, py: int) -> bool:
        """Check if a point (finger position) is within this tile."""
        return (self.x <= px < self.x + self.tile_width and
                self.y <= py < self.y + self.tile_height)

    def start_drag(self, finger_x: int, finger_y: int):
        """Start dragging the tile."""
        self.is_selected = True
        self.drag_offset_x = self.x - finger_x
        self.drag_offset_y = self.y - finger_y

    def update_drag(self, finger_x: int, finger_y: int, frame_width: int, frame_height: int):
        """Update tile position while dragging."""
        if self.is_selected:
            # Update position to follow finger
            self.x = finger_x + self.drag_offset_x
            self.y = finger_y + self.drag_offset_y

            # Keep tile within bounds
            self.x = max(0, min(self.x, frame_width - self.tile_width))
            self.y = max(0, min(self.y, frame_height - self.tile_height))

    def end_drag(self):
        """End dragging the tile."""
        self.is_selected = False

    def draw(self, frame: np.ndarray, draw_border: bool = True):
        """Draw the tile on the frame."""
        x, y = int(self.x), int(self.y)

        # Draw the tile image
        frame[y:y + self.tile_height, x:x + self.tile_width] = self.image

        if draw_border:
            # Draw border
            color = (0, 255, 0) if self.is_selected else (255, 255, 255)
            thickness = 3 if self.is_selected else 1
            cv2.rectangle(frame, (x, y),
                         (x + self.tile_width - 1, y + self.tile_height - 1),
                         color, thickness)
