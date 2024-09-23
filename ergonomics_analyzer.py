import cv2
import numpy as np
import mediapipe as mp


class ErgonomicsAnalyzer:
    """
    This class is responsible for analyzing ergonomics in a video and calculating REBA scores.
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()  # Corrected Pose initialization
        self.mp_drawing = mp.solutions.drawing_utils
        self.cycle_count = 0

    def calculate_angle(self, a, b, c):
        """
        Calculate angle between three points in 3D space.

        Args:
            a (tuple): First point coordinates (x, y, z).
            b (tuple): Second point coordinates (x, y, z).
            c (tuple): Third point coordinates (x, y, z).

        Returns:
            float: Angle in degrees.
        """
        vector1 = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
        vector2 = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def calculate_reba_score(self, angles):
        """
        Calculate REBA score based on joint angles and posture analysis.

        Args:
            angles (dict): Dictionary of joint angles such as neck, trunk, legs, arms.

        Returns:
            int: REBA score.
        """
        trunk_angle = angles.get("trunk", 0)
        neck_angle = angles.get("neck", 0)
        leg_angle = angles.get("legs", 0)

        # Basic REBA scoring logic based on posture angles (adjust based on actual REBA)
        trunk_score = 2 if trunk_angle > 60 else 1
        neck_score = 2 if neck_angle > 20 else 1
        leg_score = 2 if leg_angle > 30 else 1

        # Simplified combination to calculate total REBA score
        reba_score = trunk_score + neck_score + leg_score

        # Adjust the score further based on handling loads or repeated movements, etc.
        return reba_score

    def process_video(self, video_path):
        """
        Process the video to analyze ergonomics and calculate REBA score.

        Args:
            video_path (str): Path to the video file.
        """
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

                # Define the neck as the midpoint between the left and right shoulders
                left_shoulder = landmarks_np[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks_np[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                neck = np.mean([left_shoulder, right_shoulder], axis=0)

                # Define the trunk as the midpoint between the left and right hips
                left_hip = landmarks_np[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks_np[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                trunk = np.mean([left_hip, right_hip], axis=0)

                # Other landmarks for angles
                hip = left_hip
                knee = landmarks_np[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
                ankle = landmarks_np[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]

                # Example: Calculate angles for REBA
                neck_angle = self.calculate_angle(left_shoulder, neck, trunk)
                trunk_angle = self.calculate_angle(hip, trunk, left_shoulder)
                leg_angle = self.calculate_angle(hip, knee, ankle)

                # Store the angles
                angles = {
                    "neck": neck_angle,
                    "trunk": trunk_angle,
                    "legs": leg_angle
                }

                # Check every 10th frame (cycle)
                if self.cycle_count % 10 == 0:
                    reba_score = self.calculate_reba_score(angles)
                    print(f"Cycle {self.cycle_count}: REBA Score = {reba_score}")
                
                self.cycle_count += 1

            # Display the frame
            cv2.imshow('Pose Estimation', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Example usage:
# analyzer = ErgonomicsAnalyzer()
# analyzer.process_video("path_to_your_video.mp4")
