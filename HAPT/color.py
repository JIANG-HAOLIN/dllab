import matplotlib
import matplotlib.pyplot as plt
import numpy as np



length = 200
curve = np.linspace(1,length,length)

fig, axs = plt.subplots(2, 1, constrained_layout=True)
for ax in axs.flat:
    pcm = ax.pcolormesh(np.random.random((1,length)))


fig.colorbar(pcm, ax=[axs[1]], location='bottom')
# fig.colorbar(pcm, ax=axs[1:, :], location='right', shrink=0.6)
# fig.colorbar(pcm, ax=[axs[2, 1]], location='left')
# fig = plt.figure()
plt.plot(range(len(curve)), curve, label="data")  # plot
# ax1=plt.gca()
# ax1.patch.set_facecolor("gray")    # 设置 ax1 区域背景颜色
# ax1.patch.set_alpha(0.5)

print(curve)

plt.show()






#
#
# fig, host = plt.subplots()  #创建子图
# host.grid(False)
#
# host.set_xlim(-1, 12)
# host.set_ylim(-1, 20)
#
#
#
# x = np.linspace(0, 10, 11)
# y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1,  9.9, 13.9, 15.1, 12.5]
#
# a, b = np.polyfit(x, y, deg=1)
# y_est = a * x + b
# y_err = x.std() * np.sqrt(1/len(x) +(x - x.mean())**2 / np.sum((x - x.mean())**2))
#
#
#
# host.plot(x, y_est, '-') #连接拟合直线
#
# host.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2) #置信区间填充
#
# host.plot(x, y, 'o', color='tab:brown') #标记数据点位置
#
#
#
# plt.show()






# fig, ax = plt.subplots()
# x = np.arange(0, 4 * np.pi, 0.01)
# y = np.sin(x)
# ax.plot(x, y, color='black')
#
# threshold = 0.75
# ax.axhline(threshold, color='green', lw=2, alpha=0.7)
# ax.fill_between(x, 0, 1, where=y > threshold,
#                 color='green', alpha=0.5, transform=ax.get_xaxis_transform())
# ax.fill_between(x, 0, 1, where=(0,2),
#                 color='red', alpha=0.5, transform=ax.get_xaxis_transform())
#
# plt.show()





fig, ax = plt.subplots()
ax.plot(range(20))
ax.axvspan(8, 14, alpha=0.3, color='red',lw=0)
ax.axvspan(14, 16, alpha=0.3, color='red',lw=0)



plt.show()































