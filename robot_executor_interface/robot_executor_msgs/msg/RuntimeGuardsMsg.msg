# Volatile platform state snapshot (PR B8): the dispatch-time runtime guards
# named by the skill contracts (lease / E-stop / power / battery / faults).
# Published periodically by the executor node (~/runtime_guards); the
# planner's dispatch gate reads the latest snapshot before publishing a plan.
# Deliberately NOT planning state — volatile status stays out of contracts.

std_msgs/Header header
string robot_name
float32 battery_percent      # -1.0 = unknown (no battery state in snapshot)
bool battery_known
bool estop_pressed
bool estop_known
bool powered_on
bool power_known
bool lease_owned
bool lease_known
string lease_owner           # client name holding the body lease ("" = nobody/unknown)
string[] faults              # active behavior/system fault names
string notes
