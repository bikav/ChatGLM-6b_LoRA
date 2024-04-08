import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Step": [0, 50, 100, 150, 200, 250, 300, 350, 400, 450],
    "BLEU-4": [0.00396, 0.004193, 0.004363, 0.005466, 0.006428, 0.006686, 0.009736, 0.013715, 0.015984, 0.022105],
    "ROUGE-1": [0.137244, 0.139515, 0.144984, 0.156003, 0.171883, 0.185081, 0.191885, 0.209482, 0.220409, 0.229874],
    "ROUGE-2": [0.01775, 0.017749, 0.018589, 0.021164, 0.022443, 0.023224, 0.02474, 0.029177, 0.033896, 0.039479],
    "ROUGE-L": [0.080866, 0.080505, 0.087633, 0.098272, 0.133419, 0.144935, 0.150995, 0.163481, 0.169998, 0.177017]
}

df = pd.DataFrame(data)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plotting
metrics = ["BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
for i, metric in enumerate(metrics):
    ax = axs[i // 2, i % 2]
    ax.plot(df["Step"], df[metric], label=metric, marker='o')
    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.set_title(metric + " over Training Steps")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
