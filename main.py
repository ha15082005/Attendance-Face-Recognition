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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AttendanceSystem:
    def __init__(self):
        self.path = 'ImagesAttendance'
        self.attendance_file = 'Attendance_Records.csv'
        self.images = []
        self.classNames = []
        self.encodeListKnown = []
        self.last_face_detection_time = 0
        self.face_detection_interval = 0.1
        self.confidence_threshold = 0.5
        self.todays_attendance = set()
        self.face_history = deque(maxlen=10)
        self.current_face = None
        self.face_stable_time = 0
        self.face_stable_threshold = 0.5
        self.face_detection_interval = 0.1
        self.last_stable_face = None
        self.face_tracking_active = False
        self.tracking_timeout = 2.0

        self.load_known_faces()
        self.initialize_attendance_file()

    def initialize_attendance_file(self):
        try:
            # Check if file exists and is empty to write header, or if it does not exist at all
            write_header = not os.path.exists(self.attendance_file) or os.path.getsize(self.attendance_file) == 0

            if write_header:
                with open(self.attendance_file, mode='w', newline='', encoding='utf-8') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(['Name', 'Date', 'Time', 'Status', 'Confidence']) # Write header row
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

            attendance_key = (name, date_str) # Create unique key for this person for today

            # Check if already marked today
            if attendance_key in self.todays_attendance:
                return False # Already marked or no action needed
            self.todays_attendance.add(attendance_key) # If not marked today, add to set and append to file

            new_row = [name, date_str, time_str, 'Present', f"{confidence: .2f}"]

            # Append new row to CSV file
            with open(self.attendance_file, mode='a', newline='', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(new_row)
            logging.info(f"Attendance marked for {name} at {time_str} (Confidence: {confidence: .2f})")
            return True # Successfully marked attendance
        except Exception as e:
            logging.error(f"Error in markAttendance: {str(e)}")
            return False # Failed to mark attendance
    
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
                    logging.info("Successfully encoded a face")
                else:
                    logging.warning("No face found in image")
            except Exception as e:
                logging.error(f"Error encoding face: {str(e)}")
        return encodeList
    
    def face_detection(self, face_loc):
        self.face_history.append(face_loc) # Update history
        valid_faces = [f for f in self.face_history if f is not None] # Get recent valid face locations

        # Check if tracking is possible
        if not valid_faces:
            self.face_tracking_active = False
            self.last_stable_face = None
            return None
        
        # Calculate weighted average
        weights = np.linspace(0.5, 1.0, len(valid_faces))
        calculated_face = np.average(valid_faces, axis=0, weights=(weights / np.sum(weights))).astype(int)
        self.face_tracking_active = True
        self.last_stable_face = calculated_face
        return calculated_face
    
    def process_frame(self, img, current_time, last_tracking_time):
        draw_info = None
        if self.face_tracking_active and (current_time - last_tracking_time > self.tracking_timeout):
            logging.info("Tracking timed out")
            self.face_tracking_active = False
            self.last_stable_face = None
            self.face_history.clear()
            return None, last_tracking_time
        
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        try:
            facesCurFrame = face_recognition.face_locations(imgS, model='hog')

            if not facesCurFrame:
                self.face_detection(None)
                return None, last_tracking_time
            current_tracking_time = time.time()
            face_loc = facesCurFrame[0]
            face = self.face_detection(face_loc)

            if face is None:
                return None, current_tracking_time
            
            self.face_tracking_active = True
            y1s, x2s, y2s, x1s = face
            x1, y1, x2, y2 = int(x1s * 4), int(y1s * 4), int(x2s * 4), int(y2s * 4)

            encodesCurFrame = face_recognition.face_encodings(imgS, [face_loc])
            if not encodesCurFrame:
                logging.warning("No face encodings found")
                return None, current_tracking_time
            
            encodeFace = encodesCurFrame[0]
            name = "Unknonw"
            confidence = 0.0

            if self.encodeListKnown:
                matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if len(faceDis) > 0:
                    matchIndex = np.argmin(faceDis)
                    min_distance = faceDis[matchIndex]
                    calculated_confidence = 1 - min_distance
                    if matches[matchIndex] and calculated_confidence > self.confidence_threshold:
                        name = self.classNames[matchIndex].upper()
                        confidence = calculated_confidence
                        self.markAttendance(name, confidence)
                    else:
                        name = "Unknown"
                        confidence = calculated_confidence
            
            draw_info = (x1, y1, x2, y2, name, confidence)
            return draw_info, current_tracking_time
        except Exception as e:
            logging.error(f"Error processing frame: {str(e)}")
            self.face_tracking_active = False
            self.last_stable_face = None
            return None, current_tracking_time
        
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

        last_tracking_time = time.time()
        draw_info = None

        while True:
            success, img = cap.read()
            if not success:
                logging.error("Failed to read frame")
                break
            
            current_time = time.time()

            if current_time - self.last_face_detection_time > self.face_detection_interval:
                self.last_face_detection_time = current_time
                draw_info, last_tracking_time = self.process_frame(img, current_time, last_tracking_time)
            
            if self.face_tracking_active:
                self.draw_box(img, draw_info)
            
            cv2.imshow('Camera', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    attendance_system = AttendanceSystem()
    attendance_system.run()