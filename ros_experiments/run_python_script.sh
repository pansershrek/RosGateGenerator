# Setup enviroment
source ~/ros/sweetie_bot/devel/setup.bash

# Run python script
for x in {1..50}; do
    python3 main.py \
    --generation-logs-file=/home/panser/Desktop/RosGateGenerator/trajectories/full_trajectory_1/generation_logs_file/generation_logs_file_${x} \
    --trajectory-logs-file=/home/panser/Desktop/RosGateGenerator/trajectories/full_trajectory_1/trajectory_logs_file/trajectory_logs_file_${x};
done