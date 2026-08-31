D=/tmp/claude-1000/-media-nilum-my-stuff-Research-Human-Robot-Interaction-proactive-social-nav/d7e8a9e3-b8ba-4981-9055-07e872456c80/scratchpad
B=https://robotics.upo.es/datasets/frog/laser2d_people/data
for s in 16-41 11-36 12-43 10-31 14-57 15-53; do
  curl -sL --max-time 60 "$B/frog_${s}_odom.npz" -o $D/frog_${s}_odom.npz &
done; wait
ls -lh $D/*_odom.npz | awk '{printf "%-24s %s\n", $9, $5}'
echo "=== structure ==="
PYTHONPATH=/media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/venv/lib/python3.12/site-packages \
python3 -c "
import numpy as np
f = np.load('$D/frog_16-41_odom.npz')
print('keys:', list(f.keys()))
for k in f: print(f'  {k}: shape={f[k].shape} dtype={f[k].dtype}')
print('first 3 data rows:'); print(f['data'][:3])
print('ts[:3]:', f['ts'][:3])
"