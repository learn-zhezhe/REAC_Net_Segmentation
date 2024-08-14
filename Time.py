import time
import numpy as np

# 定义timer类，记录多种运行时间
class Timer:
    def __init__(self):
        self.times = []
        self.start()

    # 开始记录时间
    def start(self):
        self.tik = time.time()

    # 停止记录时间，并把花费的时间存储在一个列表中
    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    # 返回消耗的平均时长
    def avg(self):
        return sum(self.times) / len(self.times)

    # 返回模型运行时间总和
    def sum(self):
        return sum(self.times)

    # 返回累计时间
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()