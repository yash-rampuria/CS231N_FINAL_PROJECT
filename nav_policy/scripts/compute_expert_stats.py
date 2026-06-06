import json
from pathlib import Path
import numpy as np

ROOT = Path('nav_policy/data/eval/bc_v1')
names = ['backroom_00_sub0', 'backroom_00_sub2', 'packardpark_00_sub0']

KEY_PATH = 'path_length_expert_m'

all_v = []
all_dur = []
all_path = []
for n in names:
    er = np.load(ROOT / f'rollout_{n}' / 'expert_reference.npz')
    with open(ROOT / f'rollout_{n}' / 'metrics.json') as f:
        m = json.load(f)
    Xexp = er['Xro']
    Tro = er['Tro']
    vel = Xexp[3:6, :]
    speeds = np.linalg.norm(vel, axis=0)
    print(f'--- {n} ---')
    print(f'  duration:        {Tro[-1] - Tro[0]:.2f} s')
    print(f'  path length:     {m[KEY_PATH]:.3f} m')
    print(f'  mean speed:      {speeds.mean():.3f} m/s')
    print(f'  max speed:       {speeds.max():.3f} m/s')
    all_v.append(speeds.mean())
    all_dur.append(Tro[-1] - Tro[0])
    all_path.append(m[KEY_PATH])

print()
print('=== Aggregate ===')
print(f'  mean expert speed:    {np.mean(all_v):.3f} m/s')
print(f'  mean flight duration: {np.mean(all_dur):.2f} s')
print(f'  mean flight length:   {np.mean(all_path):.3f} m')
