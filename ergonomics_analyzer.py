import cv2
import mediapipe as mp
import numpy as np

class ErgonomicsAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points in 3D space."""
        vector1 = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
        vector2 = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame to get pose landmarks
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Draw landmarks on the frame
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Extract landmarks for further analysis
                landmarks = results.pose_landmarks.landmark
                landmarks_np = np.array([(lmk.x, lmk.y, lmk.z) for lmk in landmarks])
                
                # Example: Calculate angle between shoulder, elbow, and wrist
                shoulder = landmarks_np[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                elbow = landmarks_np[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
                wrist = landmarks_np[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                angle = self.calculate_angle(shoulder, elbow, wrist)
                print(f"Angle between shoulder, elbow, and wrist: {angle} degrees")
            
            # Display the frame
            cv2.imshow('Pose Estimation', frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'sample_video.mp4'
    analyzer = ErgonomicsAnalyzer()
    analyzer.process_video(video_path)