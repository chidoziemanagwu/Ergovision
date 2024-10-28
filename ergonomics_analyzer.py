import cv2
import numpy as np
import mediapipe as mp
import os
import webbrowser

class ErgonomicsAnalyzer:
    """
    This class is responsible for analyzing ergonomics in a video and calculating REBA scores.
    """

    def __init__(self, video_path, cycle_name):
        """
        Initialize the ErgonomicsAnalyzer.

        Args:
            video_path (str): Path to the video file.
            cycle_name (str): Name or identifier for the cycle (e.g., "Cycle 10").
        """
        self.video_path = video_path
        self.cycle_name = cycle_name
        self.mp_pose = mp.solutions.pose
        # Improved pose model initialization
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cycle_count = 0
        self.prev_angles = {"neck": 0, "trunk": 0, "legs": 0}
        self.final_reba_score = None
        self.reba_scores = []

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
        # Prevent division by zero
        if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
            return 0.0
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        # Clamp the cosine to the valid range to avoid numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def categorize_reba_score(self, score):
        """
        Categorize the REBA score based on official guidelines.

        Args:
            score (float): The REBA score.

        Returns:
            str: Category of the REBA score.
        """
        if score <= 4:
            return "Negligible Risk"
        elif 5 <= score <= 6:
            return "Low Risk"
        elif 7 <= score <= 8:
            return "Medium Risk"
        elif 9 <= score <= 10:
            return "High Risk"
        else:
            return "Very High Risk"

    def calculate_reba_score(self, angles):
        """
        Calculate REBA score based on joint angles and posture analysis.

        Args:
            angles (dict): Dictionary of joint angles such as neck, trunk, legs.

        Returns:
            int: REBA score.
        """
        neck_angle = angles.get("neck", 0)
        trunk_angle = angles.get("trunk", 0)
        leg_angle = angles.get("legs", 0)

        # REBA Scoring for Neck
        if neck_angle <= 20:
            neck_score = 0
        elif 20 < neck_angle <= 45:
            neck_score = 1
        elif 45 < neck_angle <= 90:
            neck_score = 2
        else:
            neck_score = 3

        # REBA Scoring for Trunk
        if trunk_angle <= 20:
            trunk_score = 0
        elif 20 < trunk_angle <= 45:
            trunk_score = 1
        elif 45 < trunk_angle <= 90:
            trunk_score = 2
        else:
            trunk_score = 3

        # REBA Scoring for Legs
        if leg_angle <= 30:  # Assuming leg_angle ~ knee bend; adjust thresholds as needed
            legs_score = 0
        elif 30 < leg_angle <= 60:
            legs_score = 1
        elif 60 < leg_angle <= 90:
            legs_score = 2
        else:
            legs_score = 3

        # Total REBA Score
        reba_score = neck_score + trunk_score + legs_score
        self.reba_scores.append(reba_score)

        return reba_score

    def log_posture_changes(self, current_angles):
        """
        Log posture changes if significant deviation is detected.

        Args:
            current_angles (dict): Current posture angles (neck, trunk, legs).
        """
        for joint in current_angles:
            angle_change = abs(current_angles[joint] - self.prev_angles[joint])
            if angle_change > 10:  # Log if there's a change greater than 10 degrees
                print(f"{self.cycle_name}: Significant {joint} posture change detected: {angle_change:.2f} degrees")

    def process_video(self):
        """
        Process the video to analyze ergonomics and calculate REBA score.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Error: Unable to open video file {self.video_path}")

        print(f"Processing {self.cycle_name}...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to get pose landmarks
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                # Get bounding box around the person
                image_height, image_width, _ = frame.shape
                x_coords = [lmk.x * image_width for lmk in results.pose_landmarks.landmark]
                y_coords = [lmk.y * image_height for lmk in results.pose_landmarks.landmark]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                x_max = min(image_width, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(image_height, y_max + padding)
                # Crop the frame to focus on the person
                cropped_frame = frame[y_min:y_max, x_min:x_max]

                # Draw landmarks on the cropped frame
                self.mp_drawing.draw_landmarks(cropped_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                # Extract landmarks for further analysis
                landmarks = results.pose_landmarks.landmark
                landmarks_np = np.array([(lmk.x, lmk.y, lmk.z) for lmk in landmarks])

                try:
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

                    # Calculate angles for REBA
                    neck_angle = self.calculate_angle(left_shoulder, neck, trunk)
                    trunk_angle = self.calculate_angle(hip, trunk, left_shoulder)
                    leg_angle = self.calculate_angle(hip, knee, ankle)

                    # Store the current angles
                    current_angles = {
                        "neck": neck_angle,
                        "trunk": trunk_angle,
                        "legs": leg_angle
                    }

                    # Log significant posture changes
                    self.log_posture_changes(current_angles)

                    # Update previous angles for the next frame
                    self.prev_angles = current_angles

                    # Calculate REBA score every 30 frames (adjust as needed)
                    if self.cycle_count % 30 == 0:
                        reba_score = self.calculate_reba_score(current_angles)
                        category = self.categorize_reba_score(reba_score)
                        print(f"{self.cycle_name}: Cycle {self.cycle_count}: REBA Score = {reba_score} ({category})")
                except IndexError as e:
                    print(f"Error: Unable to extract required landmarks: {e}")

            # Display the cropped frame with cycle name
            cv2.putText(cropped_frame, self.cycle_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow(self.cycle_name, cropped_frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                print(f"{self.cycle_name}: Processing interrupted by user.")
                break

            self.cycle_count += 1

        # After processing, calculate the final REBA score (average)
        if self.reba_scores:
            self.final_reba_score = round(np.mean(self.reba_scores), 2)
            category = self.categorize_reba_score(self.final_reba_score)
            print(f"{self.cycle_name}: Final REBA Score = {self.final_reba_score} ({category})")
        else:
            print(f"{self.cycle_name}: No REBA scores calculated.")

        cap.release()
        cv2.destroyWindow(self.cycle_name)


# Main code remains unchanged
if __name__ == "__main__":
    # Define the three video paths
    video_paths = [
        # ("VID1.mp4", "Cycle 10"),
        ("VID2.mp4", "Cycle 20"),
        ("VID3.mp4", "Cycle 30")
    ]

    # Create ErgonomicsAnalyzer instances for each video
    analyzers = [ErgonomicsAnalyzer(path, name) for path, name in video_paths]

    # Sequential Processing
    for analyzer in analyzers:
        analyzer.process_video()
        print(f"Finished processing {analyzer.cycle_name}. Final REBA Score: {analyzer.final_reba_score}\n")

    # Display all final REBA scores
    print("\nFinal REBA Scores:")
    for analyzer in analyzers:
        if analyzer.final_reba_score is not None:
            category = analyzer.categorize_reba_score(analyzer.final_reba_score)
            print(f"{analyzer.cycle_name}: {analyzer.final_reba_score} ({category})")
        else:
            print(f"{analyzer.cycle_name}: No REBA scores calculated.")

    # Generate HTML Report
    def generate_html_report(analyzers, output_file="reba_report.html"):
        """
        Generate an HTML report of the REBA scores.

        Args:
            analyzers (list): List of ErgonomicsAnalyzer instances.
            output_file (str): Filename for the HTML report.
        """
        # HTML Header with Bootstrap
        html_header = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>REBA Assessment Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-5">
                <h1 class="mb-4">REBA Assessment Report</h1>
                <table class="table table-striped">
                    <thead class="table-dark">
                        <tr>
                            <th scope="col">Cycle Name</th>
                            <th scope="col">Final REBA Score</th>
                            <th scope="col">Category</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        # HTML Footer
        html_footer = """
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """

        # Generate table rows
        table_rows = ""
        for analyzer in analyzers:
            if analyzer.final_reba_score is not None:
                category = analyzer.categorize_reba_score(analyzer.final_reba_score)
                table_rows += f"""
                        <tr>
                            <td>{analyzer.cycle_name}</td>
                            <td>{analyzer.final_reba_score}</td>
                            <td>{category}</td>
                        </tr>
                """
            else:
                table_rows += f"""
                        <tr>
                            <td>{analyzer.cycle_name}</td>
                            <td colspan="2">No REBA scores calculated.</td>
                        </tr>
                """

        # Combine all parts
        html_content = html_header + table_rows + html_footer

        # Write to HTML file
        with open(output_file, "w") as file:
            file.write(html_content)

        print(f"\nHTML report generated: {os.path.abspath(output_file)}")
        # Open the report in the default web browser
        webbrowser.open(f"file://{os.path.abspath(output_file)}")

    generate_html_report(analyzers)
