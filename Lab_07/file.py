import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

def load_idx_images(file_path):
  with open(file_path, 'rb') as f:
    _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
    images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
  return images / 255.0

def load_idx_labels(file_path):
  with open(file_path, 'rb') as f:
    _, num_labels = struct.unpack(">II", f.read(8))
    labels = np.frombuffer(f.read(), dtype=np.uint8)
  return labels

train_images_path = "./train-images-idx3-ubyte"
train_labels_path = "./train-labels-idx1-ubyte"
test_images_path = "./t10k-images.idx3-ubyte"
test_labels_path = "./t10k-labels.idx1-ubyte"


X_train = load_idx_images(train_images_path)
y_train = load_idx_labels(train_labels_path)
X_test = load_idx_images(test_images_path)
y_test = load_idx_labels(test_labels_path)

svm_model = SVC(kernel='rbf', gamma='scale')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
  ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
  ax.set_title(f"Pred: {y_pred[i]}")
  ax.axis('off')

plt.show()