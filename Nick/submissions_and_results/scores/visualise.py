import matplotlib.pyplot as plt
import numpy as np

# Data
languages = ['AR', 'CA', 'CS', 'DE', 'EN', 'ES', 'EU', 'FA', 'FI', 'FR', 'HI', 'IT', 'ZH']
iou_multiling = [0.3546, 0.4120, 0.2679, 0.3562, 0.3053, 0.3096, 0.3223, 0.5110, 0.3342, 0.2683, 0.4519, 0.4713, 0.1420]
iou_allvalset = [0.5371, 0.5231, 0.3936, 0.5355, 0.4721, 0.3869, 0.4570, 0.6114, 0.5514, 0.5147, 0.6572, 0.6509, 0.3201]
iou_1valset = [0.5116, 0.4215, 0.3108, 0.4799, 0.4136, 0.2970, 0.3714, 0.5632, 0.5135, 0.4513, 0.6039, 0.6414, 0.3179]

x = np.arange(len(languages))  # Language positions

# Bar width
bar_width = 0.2

# Plot
plt.figure(figsize=(15, 8))

# IoU bars
plt.bar(x - bar_width, iou_multiling, width=bar_width, label='IoU - Multiling', color='blue')
plt.bar(x, iou_allvalset, width=bar_width, label='IoU - Multiling + ALLvalset', color='green')
plt.bar(x + bar_width, iou_1valset, width=bar_width, label='IoU - Multiling + 1_Valset', color='orange')

# Labels and title
plt.xlabel('Languages', fontsize=14)
plt.ylabel('IoU Metrics', fontsize=14)
plt.title('Bert-base-multilingual-cased', fontsize=16)
plt.xticks(x, languages)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
