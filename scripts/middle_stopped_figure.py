import matplotlib.pyplot as plt

# 模型的準確性和損失數據
accuracy = [0.6040, 0.7578, 0.7863, 0.7953, 0.8383, 0.8467]
val_accuracy = [0.7749, 0.8141, 0.7542, 0.8180, 0.8182, 0.8368]
loss = [0.6514, 0.5153, 0.4701, 0.4535, 0.3819, 0.3610]
val_loss = [0.4885, 0.4346, 0.5019, 0.4570, 0.4307, 0.3997]

# 設定圖表大小
plt.figure(figsize=(12, 6))

# 繪製準確性曲線
plt.subplot(1, 2, 1)
plt.plot(range(1, 7), accuracy, label='Train Accuracy', color='blue')
plt.plot(range(1, 7), val_accuracy, label='Validation Accuracy', color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 繪製損失曲線
plt.subplot(1, 2, 2)
plt.plot(range(1, 7), loss, label='Train Loss', color='blue')
plt.plot(range(1, 7), val_loss, label='Validation Loss', color='orange')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
