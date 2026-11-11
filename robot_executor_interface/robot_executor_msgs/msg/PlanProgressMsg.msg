# One robot's progress through a plan, so the fleet can execute an ordering
# without a coordinator: every executor hears every report and gates itself.
#
# READY means "in position for this task, waiting on partners"; DONE means
# "this task is finished". Nothing here carries a time -- see ActionMsg's
# ordering fields for why.

string READY="READY"
string DONE="DONE"

std_msgs/Header header

# Matches ActionSequenceMsg.plan_id. A new plan renumbers tasks, so reports
# from a superseded plan must not open a gate in this one.
string plan_id

string robot_name
string task_id
string state
