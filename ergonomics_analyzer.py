import cv2
import numpy as np
import mediapipe as mp

class ErgonomicsAnalyzer:
    """
    This class is responsible for analyzing ergonomics in a video and calculating REBA scores.
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      model_complexity=1,
                                      enable_segmentation=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cycle_count = 0

    def calculate_angle(self, a, b, c):
        """
        Calculate angle between three points in 2D space.

        Args:
            a (tuple): First point coordinates (x, y).
            b (tuple): Second point coordinates (x, y).
            c (tuple): Third point coordinates (x, y).

        Returns:
            float: Angle in degrees.
        """
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def get_landmark_coordinates(self, landmarks, landmark_index):
        """
        Get the (x, y) coordinates of a landmark.

        Args:
            landmarks: List of landmarks.
            landmark_index: Index of the desired landmark.

        Returns:
            tuple: (x, y) coordinates.
        """
        landmark = landmarks[landmark_index]
        return (landmark.x, landmark.y)

    def map_angle_to_score(self, angle, thresholds):
        """
        Map an angle to its corresponding REBA score based on thresholds.

        Args:
            angle (float): The angle to map.
            thresholds (list): List of angle thresholds.

        Returns:
            int: REBA score.
        """
        for i, threshold in enumerate(thresholds):
            if angle <= threshold:
                return i
        return len(thresholds)

    def calculate_reba_score(self, angles):
        """
        Calculate REBA score based on joint angles and posture analysis.

        Args:
            angles (dict): Dictionary of joint angles such as neck, trunk, legs, arms.

        Returns:
            int: REBA score.
        """
        # Define angle thresholds based on REBA tables (simplified for demonstration)
        # In practice, refer to the official REBA tables for accurate scoring
        reba_scores = {}

        # Trunk
        trunk_angle = angles.get("trunk", 0)
        reba_scores['trunk'] = self.map_angle_to_score(trunk_angle, [10, 20, 30, 40, 50])

        # Neck
        neck_angle = angles.get("neck", 0)
        reba_scores['neck'] = self.map_angle_to_score(neck_angle, [10, 20, 30, 40, 50])

        # Legs
        leg_angle = angles.get("legs", 0)
        reba_scores['legs'] = self.map_angle_to_score(leg_angle, [10, 20, 30, 40, 50])

        # Upper Arm
        upper_arm_angle = angles.get("upper_arm", 0)
        reba_scores['upper_arm'] = self.map_angle_to_score(upper_arm_angle, [15, 30, 45, 60, 75])

        # Lower Arm
        lower_arm_angle = angles.get("lower_arm", 0)
        reba_scores['lower_arm'] = self.map_angle_to_score(lower_arm_angle, [15, 30, 45, 60, 75])

        # Wrist
        wrist_angle = angles.get("wrist", 0)
        reba_scores['wrist'] = self.map_angle_to_score(wrist_angle, [15, 30, 45, 60, 75])

        # Combine scores based on REBA scoring tables
        # This is a highly simplified combination. For accurate scoring, refer to the official REBA documentation.
        reba_score = sum(reba_scores.values())

        # Adjust the score based on additional factors like force, repetition, and coupling
        # These can be integrated as multipliers or additional points
        # For demonstration, let's assume:
        force = angles.get("force", 0)  # Example: 0 = none, 1 = light, 2 = moderate, 3 = high
        repetition = angles.get("repetition", 0)  # Example: 0 = none, 1 = low, 2 = moderate, 3 = high
        coupling = angles.get("coupling", 0)  # Example: 0 = none, 1 = moderate, 2 = high

        # Example adjustment (this should be based on actual REBA guidelines)
        reba_score += force + repetition + coupling

        # Ensure the REBA score is within expected range
        reba_score = min(max(reba_score, 1), 15)

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

                # Define necessary landmarks for REBA
                # Example indices based on MediaPipe's PoseLandmark enum
                LEFT_SHOULDER = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
                RIGHT_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                LEFT_ELBOW = self.mp_pose.PoseLandmark.LEFT_ELBOW.value
                RIGHT_ELBOW = self.mp_pose.PoseLandmark.RIGHT_ELBOW.value
                LEFT_WRIST = self.mp_pose.PoseLandmark.LEFT_WRIST.value
                RIGHT_WRIST = self.mp_pose.PoseLandmark.RIGHT_WRIST.value
                LEFT_HIP = self.mp_pose.PoseLandmark.LEFT_HIP.value
                RIGHT_HIP = self.mp_pose.PoseLandmark.RIGHT_HIP.value
                LEFT_KNEE = self.mp_pose.PoseLandmark.LEFT_KNEE.value
                RIGHT_KNEE = self.mp_pose.PoseLandmark.RIGHT_KNEE.value
                LEFT_ANKLE = self.mp_pose.PoseLandmark.LEFT_ANKLE.value
                RIGHT_ANKLE = self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
                NOSE = self.mp_pose.PoseLandmark.NOSE.value
                LEFT_EYE = self.mp_pose.PoseLandmark.LEFT_EYE.value
                RIGHT_EYE = self.mp_pose.PoseLandmark.RIGHT_EYE.value

                # Get coordinates
                # For simplicity, using left side. Extend to right side if needed.
                # Calculate midpoints for neck and trunk
                left_shoulder = self.get_landmark_coordinates(landmarks, LEFT_SHOULDER)
                right_shoulder = self.get_landmark_coordinates(landmarks, RIGHT_SHOULDER)
                neck = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
                trunk = self.get_landmark_coordinates(landmarks, LEFT_HIP)  # Simplified trunk position

                left_elbow = self.get_landmark_coordinates(landmarks, LEFT_ELBOW)
                left_wrist = self.get_landmark_coordinates(landmarks, LEFT_WRIST)
                left_hip = self.get_landmark_coordinates(landmarks, LEFT_HIP)
                left_knee = self.get_landmark_coordinates(landmarks, LEFT_KNEE)
                left_ankle = self.get_landmark_coordinates(landmarks, LEFT_ANKLE)

                # Calculate angles
                trunk_angle = self.calculate_angle(left_shoulder, trunk, left_hip)
                neck_angle = self.calculate_angle(left_shoulder, neck, left_elbow)
                leg_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
                upper_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                lower_arm_angle = self.calculate_angle(left_elbow, left_wrist, (left_wrist[0], left_wrist[1] + 0.1))  # Simplified
                wrist_angle = 0  # Placeholder, as wrist posture is complex to calculate

                # Placeholder values for force, repetition, coupling
                # These should be determined based on task analysis or additional input
                force = 1  # Example: light exertion
                repetition = 2  # Example: moderate repetition
                coupling = 1  # Example: moderate coupling

                # Store the angles and additional factors
                angles = {
                    "trunk": trunk_angle,
                    "neck": neck_angle,
                    "legs": leg_angle,
                    "upper_arm": upper_arm_angle,
                    "lower_arm": lower_arm_angle,
                    "wrist": wrist_angle,
                    "force": force,
                    "repetition": repetition,
                    "coupling": coupling
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
# analyzer.process_video('path_to_video.mp4')
