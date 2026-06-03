import cv2
import time
import numpy as np
from hand_tracker import HandTracker
from puzzle import PuzzleBoard

class LivePuzzle:
    def __init__(self):
        self.cap = None
        self.hand_tracker = None
        self.puzzle = None
        self.state = "WAITING"
        self.start_time = 0
        self.elapsed = 0
        self.countdown_start = 0
        self.frame_w = 640
        self.frame_h = 480
        self.puzzle_size = 300

    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.hand_tracker = HandTracker()
        self.puzzle = PuzzleBoard(3, 3)
        self.puzzle_x = (self.frame_w - self.puzzle_size) // 2
        self.puzzle_y = (self.frame_h - self.puzzle_size) // 2 + 30
        print("Live Puzzle - Press SPACE to start, Q to quit")
        print("Pinch (thumb + index) to grab tiles, drag to move, release to drop")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            canvas = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
            canvas[:] = (25, 25, 25)
            self.hand_tracker.process_frame(frame)

            if self.state == "WAITING":
                self.draw_waiting(canvas, frame)
            elif self.state == "COUNTDOWN":
                self.draw_countdown(canvas, frame)
            elif self.state == "PLAYING":
                self.draw_playing(canvas, frame)
            else:
                self.draw_complete(canvas)

            self.hand_tracker.draw_hand(canvas)
            if self.state == "PLAYING":
                self.hand_tracker.draw_cursor(canvas)

            cv2.imshow("Live Puzzle", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" ") and self.state == "WAITING":
                self.state = "COUNTDOWN"
                self.countdown_start = time.time()
            elif key == ord("r"):
                self.state = "WAITING"

        self.cap.release()
        self.hand_tracker.release()
        cv2.destroyAllWindows()

    def draw_title(self, canvas):
        text = "LIVE PUZZLE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = cv2.getTextSize(text, font, 1.0, 2)[0]
        x = (self.frame_w - size[0]) // 2
        cv2.putText(canvas, text, (x, 35), font, 1.0, (0, 215, 255), 2)

    def draw_subtitle(self, canvas, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = cv2.getTextSize(text, font, 0.4, 1)[0]
        x = (self.frame_w - size[0]) // 2
        cv2.putText(canvas, text, (x, 55), font, 0.4, (150, 150, 150), 1)

    def draw_timer(self, canvas):
        mins = self.elapsed // 60
        secs = self.elapsed % 60
        text = str(mins) + ":" + str(secs).zfill(2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = cv2.getTextSize(text, font, 0.6, 2)[0]
        x = (self.frame_w - size[0]) // 2
        pad = 8
        cv2.rectangle(canvas, (x-pad, 62), (x+size[0]+pad, 85), (50, 50, 50), -1)
        cv2.rectangle(canvas, (x-pad, 62), (x+size[0]+pad, 85), (0, 180, 220), 1)
        cv2.putText(canvas, text, (x, 80), font, 0.6, (255, 255, 255), 2)

    def draw_preview(self, canvas, frame):
        size = 80
        preview = cv2.resize(frame, (size, size))
        y = self.frame_h - size - 10
        canvas[y:y+size, 10:10+size] = preview
        cv2.rectangle(canvas, (10, y), (10+size, y+size), (80, 80, 80), 1)

    def draw_waiting(self, canvas, frame):
        self.draw_title(canvas)
        self.draw_subtitle(canvas, "Press SPACE to start | Pinch to grab, drag to move, release to drop")
        preview = cv2.resize(frame, (self.puzzle_size, self.puzzle_size))
        py, px, ps = self.puzzle_y, self.puzzle_x, self.puzzle_size
        canvas[py:py+ps, px:px+ps] = preview
        cv2.rectangle(canvas, (px, py), (px+ps, py+ps), (80, 80, 80), 2)
        self.draw_preview(canvas, frame)

    def draw_countdown(self, canvas, frame):
        elapsed = time.time() - self.countdown_start
        remaining = 3 - int(elapsed)
        if remaining <= 0:
            self.capture_puzzle(frame)
            return
        self.draw_title(canvas)
        self.draw_subtitle(canvas, "Get ready!")
        preview = cv2.resize(frame, (self.puzzle_size, self.puzzle_size))
        py, px, ps = self.puzzle_y, self.puzzle_x, self.puzzle_size
        canvas[py:py+ps, px:px+ps] = preview
        overlay = canvas[py:py+ps, px:px+ps].copy()
        overlay = (overlay * 0.4).astype(np.uint8)
        canvas[py:py+ps, px:px+ps] = overlay
        text = str(remaining)
        font = cv2.FONT_HERSHEY_SIMPLEX
        sz = cv2.getTextSize(text, font, 3, 6)[0]
        tx = px + (ps - sz[0]) // 2
        ty = py + (ps + sz[1]) // 2
        cv2.putText(canvas, text, (tx, ty), font, 3, (255, 255, 255), 6)
        self.draw_preview(canvas, frame)

    def capture_puzzle(self, frame):
        h, w = frame.shape[:2]
        size = min(h, w)
        x = (w - size) // 2
        y = (h - size) // 2
        cropped = frame[y:y+size, x:x+size]
        puzzle_img = cv2.resize(cropped, (self.puzzle_size, self.puzzle_size))
        self.puzzle.initialize_from_image(puzzle_img, self.puzzle_x, self.puzzle_y)
        self.puzzle.shuffle()
        self.start_time = time.time()
        self.state = "PLAYING"

    def draw_playing(self, canvas, frame):
        self.elapsed = int(time.time() - self.start_time)
        self.draw_title(canvas)
        self.draw_subtitle(canvas, "Pinch to grab tiles, drag to move, release to swap")
        self.draw_timer(canvas)

        cursor_pos = self.hand_tracker.get_pinch_center()
        is_pinching = self.hand_tracker.is_pinch_gesture()
        self.puzzle.handle_input(cursor_pos, is_pinching)
        self.puzzle.draw(canvas)

        if self.puzzle.is_solved:
            self.state = "COMPLETE"
        self.draw_preview(canvas, frame)

    def draw_complete(self, canvas):
        self.draw_title(canvas)
        self.puzzle.draw(canvas)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_w, self.frame_h), (0, 60, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "COMPLETE!"
        sz = cv2.getTextSize(text, font, 1.5, 3)[0]
        x = (self.frame_w - sz[0]) // 2
        cv2.putText(canvas, text, (x, self.frame_h//2 - 20), font, 1.5, (255, 255, 255), 3)
        mins = self.elapsed // 60
        secs = self.elapsed % 60
        time_text = "Time: " + str(mins) + ":" + str(secs).zfill(2)
        sz = cv2.getTextSize(time_text, font, 0.9, 2)[0]
        x = (self.frame_w - sz[0]) // 2
        cv2.putText(canvas, time_text, (x, self.frame_h//2 + 25), font, 0.9, (255, 255, 255), 2)
        hint = "Press R to restart"
        sz = cv2.getTextSize(hint, font, 0.5, 1)[0]
        x = (self.frame_w - sz[0]) // 2
        cv2.putText(canvas, hint, (x, self.frame_h//2 + 60), font, 0.5, (180, 180, 180), 1)

if __name__ == "__main__":
    game = LivePuzzle()
    game.run()