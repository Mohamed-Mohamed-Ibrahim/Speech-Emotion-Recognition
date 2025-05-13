from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np

def cnn_1d(input_len, num_classes, filters, kernel_size):
    model = models.Sequential([
        layers.Conv1D(filters[0], kernel_size=kernel_size, padding='same',
                      activation='relu', input_shape=(input_len, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(filters[1], kernel_size=kernel_size, padding='same',
                      activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        layers.Conv1D(filters[2], kernel_size=kernel_size, padding='same',
                      activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_len = X.shape[1]
num_classes = y.shape[1]
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y.argmax(1), test_size=0.2, random_state=42)

# Hyperparameters
filter_sets = [
    (32, 64, 128),
    (64, 128, 256),
    (16, 32, 64)
]
kernel_sizes = [3, 5, 7]

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Search
results = []
for filters in filter_sets:
    for k in kernel_sizes:
        print(f"\nTraining model: filters={filters}, kernel_size={k}")
        model = build_1d_cnn(input_len, num_classes, filters, k)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        val_acc = max(history.history['val_accuracy'])
        results.append(((filters, k), val_acc))
        print(f"→ Val Accuracy: {val_acc:.4f}")

# Print best model
best = sorted(results, key=lambda x: x[1], reverse=True)[0]
print(f"\n✅ Best Model: filters={best[0][0]}, kernel_size={best[0][1]} → Val Acc = {best[1]:.4f}")
