import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

        self.INDEX_TIP = 8
        self.THUMB_TIP = 4
        self.landmarks = None
        self.index_pos = None
        self.thumb_pos = None
        self.frame_w = 640
        self.frame_h = 480

        self.is_pinching = False
        self.pinch_start_dist = 40
        self.pinch_end_dist = 80

        self.pinch_release_frames = 0
        self.pinch_release_threshold = 5
        self.hand_lost_frames = 0
        self.hand_lost_threshold = 10

        self.last_index_pos = None
        self.last_thumb_pos = None

        self.thumbs_up_frames = 0
        self.thumbs_down_frames = 0
        self.gesture_confirm_frames = 15
        self.confirmed_thumbs_up = False
        self.confirmed_thumbs_down = False

        self.THUMB_MCP = 2
        self.INDEX_MCP = 5
        self.MIDDLE_TIP = 12
        self.MIDDLE_MCP = 9
        self.RING_TIP = 16
        self.RING_MCP = 13
        self.PINKY_TIP = 20
        self.PINKY_MCP = 17
        self.WRIST = 0

        self.CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17), (0, 17)
        ]

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        self.frame_w, self.frame_h = w, h
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.hand_landmarker.detect(mp_image)

        self.landmarks = None
        self.confirmed_thumbs_up = False
        self.confirmed_thumbs_down = False

        if results.hand_landmarks:
            self.hand_lost_frames = 0
            self.landmarks = results.hand_landmarks[0]
            idx = self.landmarks[self.INDEX_TIP]
            thb = self.landmarks[self.THUMB_TIP]
            self.index_pos = (int(idx.x * w), int(idx.y * h))
            self.thumb_pos = (int(thb.x * w), int(thb.y * h))
            self.last_index_pos = self.index_pos
            self.last_thumb_pos = self.thumb_pos

            dist = np.sqrt((self.index_pos[0] - self.thumb_pos[0])**2 +
                          (self.index_pos[1] - self.thumb_pos[1])**2)

            if self.is_pinching:
                if dist > self.pinch_end_dist:
                    self.pinch_release_frames += 1
                    if self.pinch_release_frames >= self.pinch_release_threshold:
                        self.is_pinching = False
                        self.pinch_release_frames = 0
                else:
                    self.pinch_release_frames = 0
            else:
                if dist < self.pinch_start_dist:
                    self.is_pinching = True
                    self.pinch_release_frames = 0

            self._detect_thumbs_gesture()
        else:
            self.hand_lost_frames += 1
            if self.last_index_pos and self.last_thumb_pos:
                self.index_pos = self.last_index_pos
                self.thumb_pos = self.last_thumb_pos
            if self.hand_lost_frames >= self.hand_lost_threshold:
                self.is_pinching = False
                self.index_pos = None
                self.thumb_pos = None
                self.thumbs_up_frames = 0
                self.thumbs_down_frames = 0

    def _detect_thumbs_gesture(self):
        if not self.landmarks:
            return
        lm = self.landmarks
        index_curled = lm[self.INDEX_TIP].y > lm[self.INDEX_MCP].y
        middle_curled = lm[self.MIDDLE_TIP].y > lm[self.MIDDLE_MCP].y
        ring_curled = lm[self.RING_TIP].y > lm[self.RING_MCP].y
        pinky_curled = lm[self.PINKY_TIP].y > lm[self.PINKY_MCP].y
        fingers_curled = index_curled and middle_curled and ring_curled and pinky_curled
        thumb_tip = lm[self.THUMB_TIP]
        wrist = lm[self.WRIST]
        thumb_extended = abs(thumb_tip.x - wrist.x) > 0.1 or abs(thumb_tip.y - wrist.y) > 0.15
        if fingers_curled and thumb_extended:
            if thumb_tip.y < wrist.y - 0.1:
                self.thumbs_up_frames += 1
                self.thumbs_down_frames = 0
            elif thumb_tip.y > wrist.y + 0.1:
                self.thumbs_down_frames += 1
                self.thumbs_up_frames = 0
            else:
                self.thumbs_up_frames = 0
                self.thumbs_down_frames = 0
        else:
            self.thumbs_up_frames = 0
            self.thumbs_down_frames = 0
        if self.thumbs_up_frames >= self.gesture_confirm_frames:
            self.confirmed_thumbs_up = True
            self.thumbs_up_frames = 0
        if self.thumbs_down_frames >= self.gesture_confirm_frames:
            self.confirmed_thumbs_down = True
            self.thumbs_down_frames = 0

    def is_thumbs_up(self):
        return self.confirmed_thumbs_up

    def is_thumbs_down(self):
        return self.confirmed_thumbs_down

    def get_pinch_center(self):
        if self.index_pos and self.thumb_pos:
            cx = (self.index_pos[0] + self.thumb_pos[0]) // 2
            cy = (self.index_pos[1] + self.thumb_pos[1]) // 2
            return (cx, cy)
        return None

    def get_index_finger_position(self):
        return self.index_pos

    def is_pinch_gesture(self):
        return self.is_pinching

    def draw_hand(self, canvas):
        if not self.landmarks:
            return
        w, h = self.frame_w, self.frame_h
        for s, e in self.CONNECTIONS:
            p1 = (int(self.landmarks[s].x * w), int(self.landmarks[s].y * h))
            p2 = (int(self.landmarks[e].x * w), int(self.landmarks[e].y * h))
            cv2.line(canvas, p1, p2, (180, 180, 180), 2)
        for i, lm in enumerate(self.landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            if i == self.INDEX_TIP or i == self.THUMB_TIP:
                color = (0, 255, 0) if self.is_pinching else (0, 255, 255)
                cv2.circle(canvas, (x, y), 6, color, -1)
            else:
                cv2.circle(canvas, (x, y), 3, (150, 150, 150), -1)

    def draw_cursor(self, canvas):
        pos = self.get_pinch_center()
        if pos:
            if self.is_pinching:
                cv2.circle(canvas, pos, 15, (0, 255, 0), -1)
                cv2.circle(canvas, pos, 15, (255, 255, 255), 2)
            else:
                cv2.circle(canvas, pos, 12, (0, 255, 255), 2)
                cv2.circle(canvas, pos, 4, (0, 255, 255), -1)

    def release(self):
        if self.hand_landmarker:
            self.hand_landmarker.close()
