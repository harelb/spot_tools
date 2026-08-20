# Since ROS doesn't support union types for messages, this message
# can represent any type of action. The action represented by this message
# is determined by the action_type field

string FOLLOW="FOLLOW"
string GAZE="GAZE"
string PICK="PICK"
string PLACE="PLACE"

string action_type

# Definition for FOLLOW
# presumably there's some other params here
nav_msgs/Path path

# Definition for GAZE
# Should gaze at the position given by gaze_pose, with the direction given by the orientation of gaze_pose.
string gaze_frame
geometry_msgs/Point robot_point
geometry_msgs/Point gaze_point

# Definition for PICK
# Should gaze at the position given by object_pose, then pick the object there with the direction given by the orientation of object_pose.
string pick_frame
string object_class
geometry_msgs/Point object_point
string object_id

# Definition for PLACE
string place_frame
# Should drop the object for now.

# --- Ordering (not timing) ---------------------------------------------
# A temporal planner's timepoints assume the durations it was given, so a slow
# pick invalidates every later one. Precedence does not decay that way, so the
# executor gates on these and never on a clock.
#
# All three are empty for a planner that produces no schedule, which leaves the
# gate inert -- the fast-downward path is unaffected.

# Scheduled task this action belongs to. Several actions share one task id: a
# relocation is a pick, a carry and a place.
string task_id

# Tasks that must be reported DONE, by any robot, before this action may start.
# Set on the first action of a task only.
string[] after_task_ids

# Robots that must be reported READY before this action may start, for a task
# allocated to a coalition. Includes this robot. Empty for a solo task.
string[] coalition_robots
