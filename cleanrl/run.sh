python sac_continuous_action.py --env-id dm_control/walker-run-v0 --num_envs 1 --smoothness caps
python sac_continuous_action.py --env-id dm_control/walker-run-v0 --num_envs 1 --smoothness 2gradcaps
python sac_continuous_action.py --env-id dm_control/walker-run-v0 --num_envs 1 --smoothness 3gradcaps

python sac_continuous_action.py --env-id dm_control/cartpole-swingup-v0 --num_envs 1 --smoothness vanilla
python sac_continuous_action.py --env-id dm_control/cartpole-swingup-v0 --num_envs 1 --smoothness caps
python sac_continuous_action.py --env-id dm_control/cartpole-swingup-v0 --num_envs 1 --smoothness gradcaps
python sac_continuous_action.py --env-id dm_control/cartpole-swingup-v0 --num_envs 1 --smoothness 2gradcaps
python sac_continuous_action.py --env-id dm_control/cartpole-swingup-v0 --num_envs 1 --smoothness 3gradcaps

python sac_continuous_action.py --env-id dm_control/ball_in_cup-catch-v0 --num_envs 1 --smoothness vanilla
python sac_continuous_action.py --env-id dm_control/ball_in_cup-catch-v0 --num_envs 1 --smoothness caps
python sac_continuous_action.py --env-id dm_control/ball_in_cup-catch-v0 --num_envs 1 --smoothness gradcaps
python sac_continuous_action.py --env-id dm_control/ball_in_cup-catch-v0 --num_envs 1 --smoothness 2gradcaps
python sac_continuous_action.py --env-id dm_control/ball_in_cup-catch-v0 --num_envs 1 --smoothness 3gradcaps

