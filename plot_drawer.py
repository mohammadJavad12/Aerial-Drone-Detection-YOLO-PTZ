import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "logs/Static_NearCenter.csv"
OUTPUT = "logs/Static_NearCenter.png"
TIME_L = 8              
YLIM_MAX = 50              

df = pd.read_csv(CSV_PATH)
df = df[df['time'] < TIME_L].reset_index(drop=True)

time = df["time"].values
error = df["angular_error"].values

first_lock = np.argmax(~np.isnan(error))

plt.figure(figsize=(10, 5))
plt.plot(time, error, "b-", linewidth=2)

plt.axvspan(time[0], time[first_lock], color="gray", alpha=0.15)
plt.text((time[0] + time[first_lock]) / 2,
         YLIM_MAX * 0.5,
         "Warmup",
         ha="center", va="center", fontsize=10, color="gray")

plt.xlim(time[0], TIME_L)
plt.ylim(0, YLIM_MAX)

plt.xlabel("Time (s)")
plt.ylabel("Angular Error (°)")
plt.title("Angular Error vs Time — Static NearCenter")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150)
plt.show()