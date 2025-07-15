# control robot to any pose
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

python run.py --mode run_arti_free_control

# control robot to open the drawer

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python run.py --mode run_arti_open