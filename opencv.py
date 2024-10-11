import cv2
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QFileDialog
from PyQt5.QtCore import QTimer, QPoint, QElapsedTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import sys
import numpy as np

# Mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

class HandDrawingApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hand Drawing App")

        # Layout setup
        layout = QVBoxLayout(self)

        # Layout for the video and canvas
        video_canvas_layout = QHBoxLayout()

        # Video label to show webcam
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        video_canvas_layout.addWidget(self.image_label)

        # Canvas label for drawing
        self.canvas_label = QLabel(self)
        self.canvas_label.setFixedSize(640, 480)
        self.canvas = QPixmap(640, 480)
        self.canvas.fill(QColor("white"))
        self.canvas_label.setPixmap(self.canvas)
        video_canvas_layout.addWidget(self.canvas_label)

        layout.addLayout(video_canvas_layout)

        # Save button
        self.save_button = QPushButton("Save Canvas as PNG", self)
        self.save_button.clicked.connect(self.save_canvas)
        layout.addWidget(self.save_button)

        # Timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Capture video from webcam
        self.capture = cv2.VideoCapture(0)

        # Variables for drawing
        self.drawing = False
        self.prev_point = None
        self.is_drawing = False  # Track if currently drawing
        self.clear_canvas_timer = QElapsedTimer()  # Timer for clearing the canvas
        self.hands_showing_time = 0

        # Start the video stream
        self.timer.start(20)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        # Flip the image to correct mirrored movement
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB for Mediapipe processing
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        # Process hand landmarks
        if results.multi_hand_landmarks:
            hands_detected = len(results.multi_hand_landmarks)

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw landmarks on the original frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Determine if the hand is right or left
                hand_label = handedness.classification[0].label

                # Get finger coordinates
                x, y = self.get_finger_coordinates(hand_landmarks)
                self.show_pencil(x, y)  # Always show the pencil position

                # Get the distance between thumb and index to adjust pen size
                pen_size = self.calculate_pen_size(hand_landmarks)

                # Check for "pen mode" (thumb and index close together)
                if self.is_pen_mode(hand_landmarks):
                    if hand_label == "Right":
                        self.draw_on_canvas(x, y, pen_size)  # Draw with right hand
                    elif hand_label == "Left":
                        self.erase_on_canvas(x, y, pen_size)  # Erase with left hand
                    self.is_drawing = True
                else:
                    self.is_drawing = False
                    self.prev_point = None  # Reset the previous point

            # Check if both hands are showing with palms facing forward for canvas clearing
            if hands_detected == 2 and self.are_palms_showing(results.multi_hand_landmarks):
                if not self.clear_canvas_timer.isValid():
                    self.clear_canvas_timer.start()
                elif self.clear_canvas_timer.elapsed() > 2000:  # 2 seconds
                    self.clear_canvas()
            else:
                self.clear_canvas_timer.invalidate()

        # Convert RGB image back to BGR for display in PyQt5
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        h, w, ch = bgr_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(bgr_image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def is_pen_mode(self, hand_landmarks):
        """Check if thumb and index finger are close together (pen holding position)."""
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Calculate the distance between thumb and index finger
        distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

        # If the distance is small enough, we are in pen mode
        return distance < 0.1  # Adjust this threshold based on your need

    def calculate_pen_size(self, hand_landmarks):
        """Calculate the pen size based on the distance between the thumb and index finger."""
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Distance between thumb and index finger in 2D space
        distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

        # Convert the distance to a suitable pen size (e.g., 1 to 20 based on the 0 to 10 cm range)
        max_distance_cm = 0.1  # ~10 cm in normalized coordinates
        pen_size = int((distance / max_distance_cm) * 20)
        pen_size = max(1, min(pen_size, 20))  # Limit pen size between 1 and 20

        return pen_size

    def get_finger_coordinates(self, hand_landmarks):
        """Get the coordinates of the index finger tip in image space."""
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x = int(index_tip.x * self.canvas_label.width())
        y = int(index_tip.y * self.canvas_label.height())
        return x, y

    def draw_on_canvas(self, x, y, pen_size):
        """Draw on the canvas based on the finger's movement."""
        painter = QPainter(self.canvas)
        pen = QPen(QColor("black"), pen_size)
        painter.setPen(pen)

        if self.prev_point is None:  # Start new line if no previous point
            self.prev_point = QPoint(x, y)
        
        painter.drawLine(self.prev_point, QPoint(x, y))
        painter.end()

        self.prev_point = QPoint(x, y)
        self.canvas_label.setPixmap(self.canvas)

    def erase_on_canvas(self, x, y, pen_size):
        """Erase on the canvas by drawing white circles where the finger moves."""
        painter = QPainter(self.canvas)
        eraser = QPen(QColor("white"), pen_size * 2)  # Thicker pen for erasing
        painter.setPen(eraser)

        if self.prev_point is None:
            self.prev_point = QPoint(x, y)
        
        painter.drawLine(self.prev_point, QPoint(x, y))
        painter.end()

        self.prev_point = QPoint(x, y)
        self.canvas_label.setPixmap(self.canvas)

    def show_pencil(self, x, y):
        """Always show pencil at the current finger location."""
        temp_canvas = self.canvas.copy()  # Copy the current canvas to draw pencil without affecting the actual canvas
        painter = QPainter(temp_canvas)
        pen = QPen(QColor("red"), 10)
        painter.setPen(pen)
        painter.drawPoint(QPoint(x, y))
        painter.end()
        self.canvas_label.setPixmap(temp_canvas)

    def are_palms_showing(self, hand_landmarks_list):
        """Check if both hands are showing palms (for clearing canvas)."""
        # Assuming the presence of hand landmarks implies palms are visible.
        # In a more complex system, you could add conditions based on palm landmarks.
        return True

    def clear_canvas(self):
        """Clear the entire canvas."""
        self.canvas.fill(QColor("white"))
        self.canvas_label.setPixmap(self.canvas)
        self.prev_point = None

    def save_canvas(self):
        """Save the canvas as a PNG file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Canvas", "", "PNG Files (*.png);;All Files (*)", options=options)
        if file_path:
            self.canvas.save(file_path, "PNG")

    def closeEvent(self, event):
        """Handle the closing of the application."""
        self.capture.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandDrawingApp()
    window.show()
    sys.exit(app.exec_())
