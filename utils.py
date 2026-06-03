"""
Utility functions for the Live Puzzle game.
"""

import cv2
import numpy as np
from typing import Tuple


def draw_title(frame: np.ndarray, title: str = "LIVE PUZZLE",
               color: Tuple[int, int, int] = (255, 255, 255),
               bg_color: Tuple[int, int, int] = (50, 50, 50)) -> np.ndarray:
    """
    Draw the game title at the top of the frame.

    Args:
        frame: Frame to draw on
        title: Title text
        color: Text color (BGR)
        bg_color: Background color (BGR)

    Returns:
        Frame with title drawn
    """
    height, width = frame.shape[:2]

    # Draw background bar
    bar_height = 50
    cv2.rectangle(frame, (0, 0), (width, bar_height), bg_color, -1)

    # Draw title text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2

    text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (bar_height + text_size[1]) // 2

    cv2.putText(frame, title, (text_x, text_y), font, font_scale, color, thickness)

    return frame


def draw_timer(frame: np.ndarray, elapsed_seconds: int,
               position: Tuple[int, int] = None,
               color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Draw the elapsed time on the frame.

    Args:
        frame: Frame to draw on
        elapsed_seconds: Elapsed time in seconds
        position: Position to draw at (default: top right)
        color: Text color (BGR)

    Returns:
        Frame with timer drawn
    """
    height, width = frame.shape[:2]

    if position is None:
        position = (width - 150, 35)

    # Format time
    minutes = elapsed_seconds // 60
    seconds = elapsed_seconds % 60
    time_str = f"Time: {minutes:02d}:{seconds:02d}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    cv2.putText(frame, time_str, position, font, font_scale, color, thickness)

    return frame


def draw_instructions(frame: np.ndarray,
                      color: Tuple[int, int, int] = (200, 200, 200)) -> np.ndarray:
    """
    Draw game instructions at the bottom of the frame.

    Args:
        frame: Frame to draw on
        color: Text color (BGR)

    Returns:
        Frame with instructions drawn
    """
    height, width = frame.shape[:2]

    instructions = [
        "Pinch (thumb + index) to grab tiles",
        "Drag to move | Release to drop",
        "Press 'R' to restart | 'Q' to quit"
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 70), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, text in enumerate(instructions):
        y = height - 55 + (i * line_height)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = (width - text_size[0]) // 2
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

    return frame


def draw_complete_screen(frame: np.ndarray, elapsed_seconds: int) -> np.ndarray:
    """
    Draw the puzzle complete screen.

    Args:
        frame: Frame to draw on
        elapsed_seconds: Total time taken

    Returns:
        Frame with complete screen drawn
    """
    height, width = frame.shape[:2]

    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 100, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw trophy emoji (as text)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Main completion text
    complete_text = "COMPLETE!"
    font_scale = 2.0
    thickness = 4
    text_size = cv2.getTextSize(complete_text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height // 2 - 30
    cv2.putText(frame, complete_text, (text_x, text_y), font, font_scale,
                (255, 255, 255), thickness)

    # Trophy symbol (using asterisks as simple representation)
    trophy_text = "* TROPHY *"
    font_scale = 1.0
    thickness = 2
    text_size = cv2.getTextSize(trophy_text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height // 2 - 80
    cv2.putText(frame, trophy_text, (text_x, text_y), font, font_scale,
                (0, 215, 255), thickness)

    # Time display
    minutes = elapsed_seconds // 60
    seconds = elapsed_seconds % 60
    time_text = f"Time: {minutes:02d}:{seconds:02d}"
    font_scale = 1.2
    thickness = 2
    text_size = cv2.getTextSize(time_text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height // 2 + 40
    cv2.putText(frame, time_text, (text_x, text_y), font, font_scale,
                (255, 255, 255), thickness)

    # Instructions
    restart_text = "Press 'R' to play again | 'Q' to quit"
    font_scale = 0.7
    thickness = 1
    text_size = cv2.getTextSize(restart_text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height // 2 + 100
    cv2.putText(frame, restart_text, (text_x, text_y), font, font_scale,
                (200, 200, 200), thickness)

    return frame


def draw_countdown(frame: np.ndarray, count: int) -> np.ndarray:
    """
    Draw a countdown number on the frame.

    Args:
        frame: Frame to draw on
        count: Countdown number

    Returns:
        Frame with countdown drawn
    """
    height, width = frame.shape[:2]

    # Draw semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Draw countdown number
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5.0
    thickness = 10

    text = str(count) if count > 0 else "GO!"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                (255, 255, 255), thickness)

    return frame


def calculate_fps(prev_time: float, curr_time: float) -> float:
    """
    Calculate frames per second.

    Args:
        prev_time: Previous frame timestamp
        curr_time: Current frame timestamp

    Returns:
        FPS value
    """
    delta = curr_time - prev_time
    if delta > 0:
        return 1.0 / delta
    return 0.0


def draw_fps(frame: np.ndarray, fps: float,
             position: Tuple[int, int] = (10, 35),
             color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw FPS counter on the frame.

    Args:
        frame: Frame to draw on
        fps: Current FPS value
        position: Position to draw at
        color: Text color (BGR)

    Returns:
        Frame with FPS drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, position, font, font_scale, color, thickness)

    return frame
