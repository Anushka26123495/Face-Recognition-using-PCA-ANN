mport numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import pickle

# Load Dataset
faces = fetch_olivetti_faces()

X = faces.data
y = faces.target

print("Dataset Shape:", X.shape)
print("Number of Classes:", len(np.unique(y)))

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardization
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA
n_components = 100

pca = PCA(n_components=n_components)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Original Features:", X.shape[1])
print("Reduced Features:", X_train_pca.shape[1])
print("Variance Retained:",
      round(np.sum(pca.explained_variance_ratio_) * 100, 2), "%")

# One-Hot Encoding
num_classes = len(np.unique(y))

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# ANN Model
model = Sequential()

model.add(Dense(128, activation='relu',
                input_shape=(n_components,)))

model.add(Dense(64, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    X_train_pca,
    y_train_cat,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# Evaluation
loss, accuracy = model.evaluate(
    X_test_pca,
    y_test_cat,
    verbose=0
)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

# Predictions
y_prob = model.predict(X_test_pca)
y_pred = np.argmax(y_prob, axis=1)

# Classification Report
print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Show Sample Predictions
plt.figure(figsize=(12, 6))

for i in range(6):
    plt.subplot(2, 3, i + 1)

    image = X_test[i].reshape(64, 64)

    plt.imshow(image, cmap='gray')

    confidence = np.max(y_prob[i]) * 100

    plt.title(
        f"Pred:{y_pred[i]}\nActual:{y_test[i]}\n{confidence:.1f}%"
    )

    plt.axis('off')

plt.tight_layout()
plt.show()

# Save Model
with open("pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)

model.save("face_recognition_ann.h5")

print("\nModel Saved Successfully")
