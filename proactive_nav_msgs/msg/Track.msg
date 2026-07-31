# One tracked person: metric position and velocity in the header frame.
int32   id          # stable track id, survives across frames
float64 x           # m
float64 y           # m
float64 vx          # m/s
float64 vy          # m/s
float64 speed       # m/s, hypot(vx, vy) — precomputed for convenience
float64 heading     # rad, atan2(vy, vx); meaningless when speed ~ 0
bool    moving      # false when the tracker judges this target stationary
