import numpy as np
import matplotlib.pyplot as plt

# 读取保存的力矩数据
torque_arr = np.load("upper_body_measured_torque_log.npy")  # shape: [steps, num_upper_dofs]

# 可选：设置时间轴（假设采样周期为 0.02s，即50Hz）
dt = 0.02
timesteps = np.arange(torque_arr.shape[0]) * dt

left_elbow_idx = 3
right_elbow_idx = 7

plt.figure(figsize=(10, 6))
# for i in range(torque_arr.shape[1]):
#     plt.plot(timesteps, torque_arr[:, i], label=f'Joint {i}')
plt.plot(timesteps, torque_arr[:, right_elbow_idx], label='Right Elbow (PD)', linewidth=2)
plt.plot(timesteps, torque_arr[:, left_elbow_idx], label='Left Elbow (CoTaP)', linewidth=2, color='red')
plt.xlabel('Time (s)', fontsize=25)
plt.ylabel('Measured Torque (Nm)', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)  # 坐标轴数字字体大小
plt.xlim([0, 10])  # 只显示前10秒
plt.title('Upper Body Joint Measured Torques', fontsize=25)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig("upper_body_measured_torque_plot.eps", format='eps', dpi=300)
plt.show()