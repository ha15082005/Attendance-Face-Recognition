import cv2
import numpy as np
import csv
import face_recognition
import os
from datetime import datetime
import pandas as pd
import time
import logging
from pathlib import Path
from collections import deque
import threading
import copy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AttendanceSystem:
    def __init__(self):
        self.path = 'ImagesAttendance'
        self.attendance_file = 'Attendance_Records.csv'
        self.images = []
        self.classNames = []
        self.encodeListKnown = []
        self.confidence_threshold = 0.5
        self.todays_attendance = set()
        self.face_history = deque(maxlen=10)
        self.load_known_faces()
        self.initialize_attendance_file()
        self.frame_to_process = None
        self.latest_result = None
        self.lock = threading.Lock()
        self.processing_thread = None
        self.running = True

    def initialize_attendance_file(self):
        try:
            write_header = not os.path.exists(self.attendance_file) or os.path.getsize(self.attendance_file) == 0
            if write_header:
                with open(self.attendance_file, mode='w', newline='', encoding='utf-8') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(['Name', 'Date', 'Time', 'Status', 'Confidence'])
                logging.info(f"Initialized new attendance file with header: {self.attendance_file}")
            else:
                logging.info(f"Attendance file already exists: {self.attendance_file}")
        except Exception as e:
            logging.error(f"Error initializing attendance file: {str(e)}")

    def markAttendance(self, name, confidence):
        try:
            now = datetime.now()
            date_str = now.strftime('%Y-%m-%d')
            time_str = now.strftime('%H:%M:%S')
            attendance_key = (name, date_str)

            if attendance_key in self.todays_attendance:
                return False
            self.todays_attendance.add(attendance_key)

            new_row = [name, date_str, time_str, 'Present', f"{confidence:.2f}"]
            with open(self.attendance_file, mode='a', newline='', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(new_row)
            logging.info(f"Attendance marked for {name} at {time_str} (Confidence: {confidence:.2f})")
            return True
        except Exception as e:
            logging.error(f"Error in markAttendance: {str(e)}")
            return False

    def load_known_faces(self):
        try:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
                logging.info(f"Created directory: {self.path}")
            myList = os.listdir(self.path)
            if not myList:
                logging.warning(f"No images found in {self.path}")
                return

            for cl in myList:
                try:
                    curImg = cv2.imread(f'{self.path}/{cl}')
                    if curImg is None:
                        logging.error(f"Failed to load image: {cl}")
                        continue
                    self.images.append(curImg)
                    self.classNames.append(os.path.splitext(cl)[0])
                except Exception as e:
                    logging.error(f"Error loading image {cl}: {str(e)}")

            if self.images:
                self.encodeListKnown = self.findEncodings(self.images)
                logging.info(f"Encoding Complete. Found {len(self.encodeListKnown)} face encodings.")
            else:
                logging.warning("No valid images were loaded")
        except Exception as e:
            logging.error(f"Error in load_known_faces: {str(e)}")

    def findEncodings(self, images):
        encodeList = []
        for img in images:
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(img, model='hog')
                if face_locations:
                    encode = face_recognition.face_encodings(img, face_locations)[0]
                    encodeList.append(encode)
                else:
                    logging.warning("No face found in image used for encoding")
            except Exception as e:
                logging.error(f"Error encoding face: {str(e)}")
        return encodeList

    def face_detection(self, face_loc):
        self.face_history.append(face_loc)
        valid_faces = [f for f in self.face_history if f is not None]

        if not valid_faces:
            return None
        
        avg_face = np.mean(valid_faces, axis=0).astype(int)
        return tuple(avg_face)

    def process_frame(self):
        logging.info("Processing thread started.")
        last_processed_name = None
        time_since_last_known = 0

        while self.running:
            frame_copy = None
            with self.lock:
                if self.frame_to_process is not None:
                    frame_copy = self.frame_to_process.copy()
                    self.frame_to_process = None

            if frame_copy is None:
                time.sleep(0.01)
                continue

            imgS = cv2.resize(frame_copy, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            current_result = None

            try:
                facesCurFrame = face_recognition.face_locations(imgS, model='hog')
                if facesCurFrame:
                    face_loc = facesCurFrame[0] 
                    face = self.face_detection(face_loc) 
                    if face:
                        encodesCurFrame = face_recognition.face_encodings(imgS, [face_loc]) 
                        if encodesCurFrame:
                            encodeFace = encodesCurFrame[0]
                            name = "Unknown"
                            confidence = 0.0
                            if self.encodeListKnown:
                                faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                                if len(faceDis) > 0:
                                    matchIndex = np.argmin(faceDis)
                                    min_distance = faceDis[matchIndex]
                                    if min_distance < (1.0 - self.confidence_threshold):
                                        name = self.classNames[matchIndex].upper()
                                        confidence = 1 - min_distance
                                        if name != last_processed_name or time.time() - time_since_last_known > 5:
                                            if self.markAttendance(name, confidence):
                                                last_processed_name = name
                                                time_since_last_known = time.time()
                                    else:
                                        name = "Unknown"
                                        confidence = 1 - min_distance
                                        last_processed_name = "Unknown"
                            
                            y1s, x2s, y2s, x1s = face
                            x1, y1, x2, y2 = int(x1s * 4), int(y1s * 4), int(x2s * 4), int(y2s * 4)
                            current_result = (x1, y1, x2, y2, name, confidence)
                        else:
                            self.face_detection(None) 
                            last_processed_name = None
                    else:
                        last_processed_name = None
                else:
                    self.face_detection(None)
                    last_processed_name = None
            except Exception as e:
                logging.error(f"Error processing frame in thread: {str(e)}")
                self.face_detection(None)
                last_processed_name = None

            with self.lock:
                self.latest_result = current_result

        logging.info("Processing thread finished.")

    def draw_box(self, img, draw_info):
        if draw_info is None:
            return

        try:
            x1, y1, x2, y2, name, confidence = draw_info
            if name == "Unknown":
                box_color = (0, 0, 255)
                text = "Unknown"
            else:
                box_color = (0, 255, 0)
                text = f"{name} ({confidence:.2f})"

            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), box_color, cv2.FILLED)
            cv2.putText(img, text, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

        except Exception as e:
            logging.error(f"Error drawing box: {str(e)}")

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Failed to open camera")
            return
        logging.info("Camera opened successfully")

        self.running = True
        self.processing_thread = threading.Thread(target=self.process_frame, daemon=True)
        self.processing_thread.start()

        current_draw_info = None

        while True:
            success, img = cap.read()
            if not success:
                logging.error("Failed to read frame")
                break

            with self.lock:
                self.frame_to_process = img

            with self.lock:
                if self.latest_result is not None:
                    current_draw_info = copy.deepcopy(self.latest_result)

            if current_draw_info:
                self.draw_box(img, current_draw_info)

            cv2.imshow('Camera', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        logging.info("Stopping threads...")
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Application finished.")


if __name__ == "__main__":
    attendance_system = AttendanceSystem()
    attendance_system.run()