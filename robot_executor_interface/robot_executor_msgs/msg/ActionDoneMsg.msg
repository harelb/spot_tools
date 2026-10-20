# One action a robot finished, successful or not, whichever planner made it.
#
# Mission progress -- which places were reached, which objects inspected, picked
# or put down -- is read from these, so it is the same for every planner.
# Published once per action, after its final attempt.

std_msgs/Header header

# Matches ActionSequenceMsg.plan_id.
string plan_id
string robot_name

# One of ActionMsg's FOLLOW / GAZE / PICK / PLACE.
string action_type
bool success

# GAZE / PICK / PLACE: the object acted on.
string object_id

# FOLLOW: the waypoints driven, in order; PLACE: where the object was put down.
geometry_msgs/Point[] points
