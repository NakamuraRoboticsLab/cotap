<h1 align="center"> FALCON: Learning Force-Adaptive Humanoid Loco-Manipulation </h1>

<div align="center">

<!-- Robotics: Science and Systems (RSS) 2025 -->

[[Website]](https://lecar-lab.github.io/falcon-humanoid/)
[[Arxiv]](https://lecar-lab.github.io/falcon-humanoid/)
[[Video]](https://www.youtube.com/watch?v=OfsvJ5-Fyzg)

<img src="assets/ip.png" style="height:100px;" />




[![IsaacGym](https://img.shields.io/badge/IsaacGym-Preview4-b.svg)](https://developer.nvidia.com/isaac-gym) [![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg)](https://ubuntu.com/blog/tag/22-04-lts) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()


<img src="assets/hero_static.gif" width="100%"/>

</div>

## TODO
- [x] Release training code
- [x] Release sim2sim code
- [x] Release sim2real code
- [ ] Compatible with IsaacSim
- [ ] Compatible with Genesis


# Installation

## IsaacGym Conda Env

Create mamba/conda environment, in the following we use conda for example, but you can use mamba as well.

```bash
conda create -n fcgym python=3.8
conda activate fcgym
```
### Install IsaacGym

Download [IsaacGym](https://developer.nvidia.com/isaac-gym/download) and extract:

```bash
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
```

Install IsaacGym Python API:

```bash
pip install -e isaacgym/python
```

Test installation:

```bash
cd isaacgym/python/examples

python 1080_balls_of_solitude.py  # or
python joint_monkey.py
```

For libpython error:

- Check conda path:
    ```bash
    conda info -e
    ```
- Set LD_LIBRARY_PATH:
    ```bash
    export LD_LIBRARY_PATH=</path/to/conda/envs/your_env/lib>:$LD_LIBRARY_PATH
    ```

### Install FALCON

```bash
git clone https://github.com/LeCAR-Lab/FALCON.git
cd FALCON

pip install -e .
pip install -e isaac_utils
```

# Motion Retargetting
Please refer to [PHC](https://github.com/ZhengyiLuo/PHC).

# FALCON Training
## Unitree G1_29DoF
<details>
<summary>Training Command</summary>

```bash
python humanoidverse/train_agent.py \
+exp=decoupled_locomotion_stand_height_waist_wbc_diff_force_ma_ppo_ma_env \
+simulator=isaacgym \
+domain_rand=domain_rand_rl_gym \
+rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force \
+robot=g1/g1_29dof_waist_fakehand \
+terrain=terrain_locomotion_plane \
+obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma \
num_envs=4096 \
project_name=g1_29dof_falcon \
experiment_name=g1_29dof_falcon \
+opt=wandb \
obs.add_noise=True \
env.config.fix_upper_body_prob=0.3 \
robot.dof_effort_limit_scale=0.9 \
rewards.reward_initial_penalty_scale=0.1 \
rewards.reward_penalty_degree=0.0001
```

```bash
python humanoidverse/train_agent.py \
+exp=decoupled_locomotion_stand_height_waist_wbc_diff_force_ma_ppo_ma_env \
+simulator=isaacgym \
+domain_rand=domain_rand_rl_gym \
+rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force \
+robot=g1/g1_29dof_waist_fakehand \
+terrain=terrain_locomotion_plane \
+obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma \
num_envs=4096 \
project_name=g1_29dof_falcon \
experiment_name=g1_29dof_falcon \
obs.add_noise=True \
env.config.fix_upper_body_prob=0.3 \
robot.dof_effort_limit_scale=0.9 \
rewards.reward_initial_penalty_scale=0.1 \
rewards.reward_penalty_degree=0.0001
```

</details>

<details>
<summary>Evaluation Command</summary>

```bash
python humanoidverse/eval_agent.py \
+checkpoint=<path_to_your_ckpt>
```
</details>


After around `6k` iterations, in `IsaacGym`:
<table>
  <tr>
    <td style="text-align: center;">
      <img src="assets/g1.gif" style="width: 100%;"/>
    </td>
  </tr>
</table>

## Booster T1_29DoF
<details>
<summary>Training Command</summary>

```bash
python humanoidverse/train_agent.py \
+exp=decoupled_locomotion_stand_height_waist_wbc_diff_force_ma_ppo_ma_env \
+simulator=isaacgym \
+domain_rand=domain_rand_rl_gym \
+rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force \
+robot=t1/t1_29dof_waist_wrist \
+terrain=terrain_locomotion_plane \
+obs=dec_loco/t1_29dof_obs_diff_force_history_wolinvel_ma \
num_envs=4096 \
project_name=t1_29dof_falcon \
experiment_name=t1_29dof_falcon \
+opt=wandb \
obs.add_noise=True \
env.config.fix_upper_body_prob=0.3 \
robot.dof_effort_limit_scale=0.9 \
rewards.reward_initial_penalty_scale=0.1 \
rewards.reward_penalty_degree=0.0001 \
rewards.feet_height_target=0.08 \
rewards.feet_height_stand=0.02 \
rewards.desired_feet_max_height_for_this_air=0.08 \
rewards.desired_base_height=0.62 \
rewards.reward_scales.penalty_lower_body_action_rate=-0.5 \
rewards.reward_scales.penalty_upper_body_action_rate=-0.5 \
env.config.apply_force_pos_ratio_range=[0.5,2.0]
```

</details>

## Unitree H1_19DoF
<details>
<summary>Training Command</summary>

```bash
python humanoidverse/train_agent.py \
+exp=decoupled_locomotion_stand_height_waist_wbc_diff_force_ma_ppo_ma_env \
+simulator=isaacgym \
+domain_rand=domain_rand_rl_gym \
+rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force \
+robot=h1/h1 \
+terrain=terrain_locomotion_plane \
+obs=dec_loco/h1_19dof_obs_diff_force_history_wolinvel_ma \
num_envs=4096 \
project_name=h1_19dof_falcon \
experiment_name=h1_19dof_falcon \
obs.add_noise=True \
rewards.desired_base_height=0.9 \
env.config.fix_upper_body_prob=0.3 \
env.config.apply_force_pos_ratio_range=[0.5,2.0]
```

</details>

<details>
<summary>Evaluation Command</summary>

```bash
python humanoidverse/eval_agent.py \
+checkpoint=<path_to_your_ckpt>
```
</details>


After around `6k` iterations, in `IsaacGym`:
<table>
  <tr>
    <td style="text-align: center;">
      <img src="assets/t1.gif" style="width: 100%;"/>
    </td>
  </tr>
</table>

# FALCON Deploy
We provide seamless sim2sim and sim2real deployment scripts supporting both [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) and [booster_robotics_sdk](https://github.com/hang0610/booster_robotics_sdk). Please refer to this [README](sim2real/README.md) for details.

<table>
  <tr>
    <td style="text-align: center;">
      <img src="assets/deploy.png" style="width: 99%;"/>
    </td>
  </tr>
</table>

# FALCON Extension

## Large Workspace
FALCON can be easily extended to larger workspace by setting larger torso command range and base height command range. We provide the sim2sim result of Unitree G1 with larger command range as an example:

https://github.com/user-attachments/assets/2d92000e-b990-45aa-a4ad-032fe0158eba


# Citation
If you find our work useful, please consider citing us!

```bibtex
@article{zhang2025falcon,
          title={FALCON: Learning Force-Adaptive Humanoid Loco-Manipulation},
          author={Zhang, Yuanhang and Yuan, Yifu and Gurunath, Prajwal and He, Tairan and Omidshafiei, Shayegan and Agha-mohammadi, Ali-akbar and Vazquez-Chanlatte, Marcell and Pedersen, Liam and Shi, Guanya},
          journal={arXiv preprint arXiv:2505.06776},
          year={2025}
        }
```

Other work also using FALCON's dual-agent framework:

```bibtex
@article{li2025softa,
          title={Hold My Beer: Learning Gentle Humanoid Locomotion and End-Effector Stabilization Control},
          author={Li, Yitang and Zhang, Yuanhang and Xiao, Wenli and Pan, Chaoyi and Weng, Haoyang and He, Guanqi and He, Tairan and Shi, Guanya},
          journal={arXiv preprint arXiv:2505.24198},
          year={2025}
        }
```

# Acknowledgement
**FALCON** is built upon [ASAP](https://github.com/LeCAR-Lab/ASAP) and [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse).

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
