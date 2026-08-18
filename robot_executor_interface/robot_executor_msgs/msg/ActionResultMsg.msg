# Result of ONE executed action within a dispatched sequence (PR B5):
# the executor -> planner return channel. Published by the executor node
# (~/action_result) as each action of an ActionSequenceMsg completes, is
# retried out, or is pre-empted; consumed by omniplanner's robot adaptors.

string STATUS_SUCCESS="SUCCESS"
string STATUS_FAILED="FAILED"
string STATUS_PREEMPTED="PREEMPTED"

std_msgs/Header header
string robot_name          # ActionSequenceMsg.robot_name of the dispatched sequence
string plan_id             # ActionSequenceMsg.plan_id of the dispatched sequence
string action_type         # ActionMsg action_type (FOLLOW / GAZE / PICK / PLACE)
uint32 action_index        # position within the dispatched sequence
string status              # one of STATUS_*
string detail              # free-text notes (failure reason, retry count, ...)
