import cv2
import matplotlib.pyplot as plt
import shapely


class FeedbackCollector:
    def __init__(self):
        plt.ion()

        plt.figure(2)
        self.progress_scat = plt.scatter([], [], color="r")
        self.goal_scat = plt.scatter([], [], color="g")

    def print(self, level, s):
        print(level + " " + str(s))

    def pick_image_feedback(self, semantic_image, mask_image):
        plt.figure(0)
        plt.imshow(semantic_image)

        plt.figure(1)
        plt.imshow(mask_image)

    def follow_path_feedback(self, path):
        plt.figure(2)
        plt.plot(path[:, 0], path[:, 1])
        plt.pause(0.1)

    def path_following_progress_feedback(
        self, progress_point: shapely.Point, target_point: shapely.Point
    ):
        plt.figure(2)

        self.progress_scat.set_offsets([progress_point.x, progress_point.y])
        self.goal_scat.set_offsets([target_point.x, target_point.y])
        plt.pause(0.1)

    def gaze_feedback(self, current_pose, gaze_point):
        plt.figure(2)
        plt.plot(
            [current_pose[0], gaze_point[0]],
            [current_pose[1], gaze_point[1]],
            color="b",
        )
        plt.pause(0.1)

    def bounding_box_detection_feedback(
        self, annotated_img, centroid_x, centroid_y, semantic_class
    ):
        # Draw bounding box and label
        # cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if centroid_x is None or centroid_y is None:
            pass

        else:
            label = f"{semantic_class}"
            cv2.putText(
                annotated_img,
                label,
                (centroid_x, centroid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Label the centroid
            cv2.circle(annotated_img, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
            cv2.putText(
                annotated_img,
                "Centroid",
                (centroid_x + 10, centroid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        # Display or save the annotated image
        cv2.imshow("Most Confident Output", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def report_action_done(self, sequence, command, success):
        print(
            f"[ACTION DONE] {type(command).__name__} "
            f"{getattr(command, 'object_id', '')} success={success}"
        )

    def set_robot_holding_state(
        self, is_holding: bool, object_id: str, position=None, timeout=5
    ):
        action = "picked up" if is_holding else "placed down"
        print(
            f"[HOLDING STATE] Robot {action} object '{object_id}' (is_holding={is_holding})"
        )
        return True
