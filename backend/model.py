#!/usr/bin/env python
# coding: utf-8

# # Garbage Classification Using Transfer Learning
# ---------------
# Developer : Shanmugaraj

# # 1 - Importing Libraries

# In[2]:


import os
import numpy as np  # Importing NumPy for numerical operations and array manipulations
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting graphs and visualizations
import seaborn as sns  # Importing Seaborn for statistical data visualization, built on top of Matplotlib
import tensorflow as tf  # Importing TensorFlow for building and training machine learning models
from tensorflow import keras  # Importing Keras, a high-level API for TensorFlow, to simplify model building
from tensorflow.keras import Layer  # Importing Layer class for creating custom layers in Keras
from tensorflow.keras.models import Sequential  # Importing Sequential model for building neural networks layer-by-layer
from tensorflow.keras.layers import Rescaling , GlobalAveragePooling2D
from tensorflow.keras import layers, optimizers, callbacks  # Importing various modules for layers, optimizers, and callbacks in Keras
from sklearn.utils.class_weight import compute_class_weight  # Importing function to compute class weights for imbalanced datasets
from tensorflow.keras.applications import EfficientNetV2B2  # Importing EfficientNetV2S model for transfer learning
from sklearn.metrics import confusion_matrix, classification_report  # Importing functions to evaluate model performance
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#Loading Dataset
dataset_dir = os.getenv("DATASET_DIR")

if dataset_dir is None:
    print("⚠️ WARNING: DATASET_DIR environment variable is missing!")
    # Fallback or exit
    dataset_dir = "TrashType_Image_Dataset" # Default fallback for local dev if .env is missing but folder exists


image_size = (224, 224)
batch_size = 32
seed = 42

# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    shuffle=True,
    image_size=image_size,
    batch_size=batch_size
)

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    shuffle=True,
    image_size=image_size,
    batch_size=batch_size
)

# Get class names from validation dataset
val_class = val_ds.class_names

# Split validation into test and validation sets
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 2)
val_dat = val_ds.skip(val_batches // 2)

# Optimize test dataset
test_ds_eval = test_ds.cache().prefetch(tf.data.AUTOTUNE)

# ✨ Pretty output section
print("\nDataset Summary")
print("-" * 30)

print("Class Names:")
for i, cls in enumerate(train_ds.class_names, 1):
    print(f"  {i}. {cls}")

print(f"\nTotal Classes     : {len(train_ds.class_names)}")
print(f"Training Batches  : {len(train_ds)}")
print(f"Validation Batches: {len(val_dat)}")
print(f"Test Batches      : {len(test_ds)}")
print("-" * 30)


# # 3 - Visualizing the Dataset

# In[4]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    num_images = min(len(images), 12)
    for i in range(num_images):
        ax = plt.subplot(4, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"{train_ds.class_names[labels[i]]} ({i})", fontsize=10)
        plt.axis("off")

plt.suptitle("Sample Images from Training Dataset", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # adjust to make space for title
plt.show()


# # 4 - Enhanced Class Distribution Visualization

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Function to count class distribution (in %)
def count_distribution(dataset, class_names):
    total = 0
    counts = {name: 0 for name in class_names}
    
    for _, labels in dataset:
        for label in labels.numpy():
            counts[class_names[label]] += 1
            total += 1

    return {k: round((v / total) * 100, 2) for k, v in counts.items()}

# Improved bar plot function with seaborn
def simple_bar_plot(dist, title):
    plt.figure(figsize=(9, 6))
    sns.set_style("whitegrid")
    
    bars = plt.bar(dist.keys(), dist.values(), color=sns.color_palette("pastel"), edgecolor='black')

    # Add value labels on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

# Get class names
class_names = train_ds.class_names

# Compute distributions
train_dist = count_distribution(train_ds, class_names)
val_dist = count_distribution(val_ds, class_names)
test_dist = count_distribution(test_ds, class_names)
overall_dist = {k: round((train_dist[k] + val_dist[k]) / 2, 2) for k in class_names}

# Create DataFrame
dist_df = pd.DataFrame({
    'Class': class_names,
    'Train (%)': [train_dist[k] for k in class_names],
    'Validation (%)': [val_dist[k] for k in class_names],
    'Test (%)': [test_dist[k] for k in class_names],
    'Overall (%)': [overall_dist[k] for k in class_names],
})

# Print clean table
print("\nClass Distribution Summary:\n")
print(dist_df.to_string(index=False))

# Plot distributions
simple_bar_plot(train_dist, "Training Set Class Distribution (%)")
simple_bar_plot(val_dist, "Validation Set Class Distribution (%)")
simple_bar_plot(test_dist, "Test Set Class Distribution (%)")
simple_bar_plot(overall_dist, "Overall Class Distribution (%)")


# # 5 - Calculating Class Weights to Handle Imbalance

# In[6]:


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Count class occurrences and collect all labels
class_counts = {i: 0 for i in range(len(class_names))}
all_labels = []

for images, labels in train_ds:
    for label in labels.numpy():
        class_counts[label] += 1
        all_labels.append(label)

# Compute class weights (balanced)
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(class_names)),
    y=all_labels
)

# Map class index to weight
class_weights = {i: round(w, 4) for i, w in enumerate(class_weights_array)}

# ✨ Display results nicely
print("\n📊 Class Distribution in Training Set")
print("-" * 40)
for i, name in enumerate(class_names):
    print(f"{i}. {name:<10} ➤ Samples: {class_counts[i]:<4} | Weight: {class_weights[i]}")
print("-" * 40)
print(f"\n🧮 Total Training Samples: {sum(class_counts.values())}")



# # 6 - Preprocessing and Model Training

# In[11]:


# Unified Model Training Pipeline with 128x128 Image Input and Enhanced Output

import pandas as pd
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetV2B2
import tensorflow as tf
import pickle

# Data Augmentation
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Load Pretrained EfficientNetV2B2 with matching input size
base_model = EfficientNetV2B2(
    include_top=False,
    input_shape=(224, 224, 3),
    include_preprocessing=True,
    weights='imagenet'
)

# Freeze early layers
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

# Build Final Model
model = Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    base_model,
    GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(6, activation='softmax')
])

# Compile Model
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    filepath='best_model224.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Pre-training Summary
epochs = 15
print("\n📦 Starting Model Training...")
print(f"Epochs       : {epochs}")
print(f"Class Weights: {class_weights}")
print(f"Checkpoint   : Best model saved to 'best_model224.keras'")
print(f"Early Stop   : Enabled with patience=3\n")

# Train the Model with Tqdm progress bar
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights,
    batch_size=32,
    callbacks=[early, checkpoint, TqdmCallback(verbose=1)]
)

# Display training log as a table
df_log = pd.DataFrame(history.history)
print("\n📊 Training Log Summary:")
print(df_log.to_string(index=True))

# Save history for later comparison
with open("history224.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Plot Accuracy and Loss Graphs
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# In[12]:


from tensorflow.keras.models import load_model

model = load_model("best_model224.keras")
model.summary()


# In[13]:


# Find the base model layer by layer name or type
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "efficientnetv2b2" in layer.name.lower():
        base_model = layer
        break

# Print summary of the extracted base model
base_model.summary()


# # 7 - Model Evaluation

# In[16]:


from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

def evaluate_model(model_path, test_dataset, class_names):
    # Load the model
    print(f"\nLoading Model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Evaluate on test set
    print("\nEvaluating Model on Test Dataset...\n" + "-"*50)
    loss, accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"✅ Test Accuracy : {accuracy:.4f}")
    print(f"📉 Test Loss     : {loss:.4f}")
    print("-"*50)

    # True and predicted labels
    y_true = np.concatenate([labels.numpy() for _, labels in test_dataset])
    y_pred_probs = model.predict(test_dataset, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification Report
    print("\n📊 Classification Report:\n" + "-"*50)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    print("-"*50)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5,
                xticklabels=class_names, yticklabels=class_names)

    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 🧪 Example usage
evaluate_model("best_model224.keras", test_ds_eval, class_names)


# # 8 - Final Testing

# In[18]:


import matplotlib.pyplot as plt
import tensorflow as tf

# Load your trained model
model_path = 'best_model224.keras'  # Change if needed
model = tf.keras.models.load_model(model_path)

# Get class names from training dataset
class_names = train_ds.class_names  

# Shuffle test dataset before sampling
test_ds_eval_shuffled = test_ds_eval.shuffle(1000, reshuffle_each_iteration=True)

# Take one random batch from test dataset and predict
for images, labels in test_ds_eval_shuffled.take(1):  
    predictions = model.predict(images)  
    pred_labels = tf.argmax(predictions, axis=1)

    # Plot first 8 images with predictions
    plt.figure(figsize=(18, 8))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        true_cls = class_names[labels[i]]
        pred_cls = class_names[pred_labels[i]]
        
        # Green if correct, red if wrong
        title_color = 'green' if true_cls == pred_cls else 'red'
        plt.title(f"True: {true_cls}\nPred: {pred_cls}", color=title_color, fontsize=14, fontweight='bold')  # ← increased font size here
        plt.axis("off")
    
    plt.suptitle(f"Model Predictions from '{model_path}'", fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# # 9 - Retraining Model

# In[19]:


#  Retraining Pipeline with Accuracy Comparison and Conditional Saving + Custom CNN Layers

import pandas as pd
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, layers, models
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

#  Load the previously saved model
old_model_path = 'best_model224.keras'
model_save_path = 'best_model_finetuned224.keras'

old_model = tf.keras.models.load_model(old_model_path)

#  Evaluate old model on validation set
val_images, val_labels = [], []
for x, y in val_ds.unbatch():
    val_images.append(x.numpy())
    val_labels.append(y.numpy())

val_images = np.stack(val_images)
val_labels = np.array(val_labels)

old_preds = old_model.predict(val_images, verbose=0)
old_pred_labels = np.argmax(old_preds, axis=1)
old_val_accuracy = accuracy_score(val_labels, old_pred_labels)
print(f"📏 Old Model Accuracy: {old_val_accuracy*100:.2f}%")

#  Extract the EfficientNetV2B2 base model from the saved model
base_model = None
for layer in old_model.layers:
    if isinstance(layer, tf.keras.Model) and 'efficientnet' in layer.name:
        base_model = layer
        break
if base_model is None:
    raise ValueError("EfficientNetV2B2 base model not found inside loaded model.")

#  Unfreeze some deeper layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:75]:
    layer.trainable = False

#  Create new model with added custom CNN layers
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    base_model,
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(6, activation='softmax')
])

#  Compile the model
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#  Custom callback to conditionally save improved models
class ConditionalSave(tf.keras.callbacks.Callback):
    def __init__(self, initial_best, filename):
        self.best_accuracy = initial_best
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get("val_accuracy")
        if val_acc and val_acc > self.best_accuracy:
            print(f"\n💾 Improved! New Val Acc: {val_acc:.4f} > Prev Best: {self.best_accuracy:.4f}. Saving as '{self.filename}'...")
            self.model.save(self.filename)
            self.best_accuracy = val_acc

#  Callbacks
early = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

#  Fine-tuning Info
epochs = 15
print("\n🔁 Fine-Tuning EfficientNetV2B2 + Custom CNN...")
print(f"🔓 Unfroze layers after 75")
print(f"🎯 Previous Validation Accuracy: {old_val_accuracy*100:.2f}%")
print(f"💾 Will save only if validation accuracy improves over previous\n")

#  Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights,
    batch_size=32,
    callbacks=[
        early,
        TqdmCallback(verbose=1),
        ConditionalSave(initial_best=old_val_accuracy, filename=model_save_path)
    ]
)

#  Save history
with open("history_finetuned224.pkl", "wb") as f:
    pickle.dump(history.history, f)

#  Show Log Table
df_log = pd.DataFrame(history.history)
print("\n📊 Fine-tuning Log Summary:")
print(df_log.to_string(index=True))

#  Plot Training Curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
plt.title('Fine-tuning Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Fine-tuning Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# # 10 - Evaluate Model Matrics

# In[25]:


import tensorflow as tf
import pickle
import pandas as pd

# === Paths ===
model_path = 'best_model_finetuned224.keras'     
history_path = 'history_finetuned224.pkl'         

# === Load the model ===
model = tf.keras.models.load_model(model_path)
print(f"Loaded model from: {model_path}")


# === Load history ===
with open(history_path, 'rb') as f:
    history = pickle.load(f)

# === Get final epoch metrics ===
final_accuracy     = history['accuracy'][-1]
final_val_accuracy = history['val_accuracy'][-1]
final_loss         = history['loss'][-1]
final_val_loss     = history['val_loss'][-1]

print("\n📊 Final Training Summary:")
print(f"Train Accuracy     : {final_accuracy:.4f}")
print(f"Validation Accuracy: {final_val_accuracy:.4f}")
print(f"Train Loss         : {final_loss:.4f}")
print(f"Validation Loss    : {final_val_loss:.4f}")


# # 11 - Comparing Model Performance: Accuracy & Loss Trends

# In[24]:


import matplotlib.pyplot as plt
import pickle

def plot_model_history(history_file, model_name="Model"):
    # Load training history from a .pkl file
    with open(history_file, "rb") as f:
        history = pickle.load(f)

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs_range = range(len(acc))

    # Set up plots
    plt.figure(figsize=(12, 5))

    # --- Accuracy Plot ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Acc', color='green', marker='o')
    plt.plot(epochs_range, val_acc, label='Val Acc', color='blue', marker='s', linestyle='--')
    plt.title(f'📈 Accuracy - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.xticks(epochs_range)

    # --- Loss Plot ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss', color='red', marker='o')
    plt.plot(epochs_range, val_loss, label='Val Loss', color='orange', marker='s', linestyle='--')
    plt.title(f'📉 Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.xticks(epochs_range)

    # Layout & show
    plt.suptitle(f"📊 Training Summary for {model_name}", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()


# In[29]:


plot_model_history("history_finetuned224.pkl", model_name="Retrained Model 224 with Custom CNN Layers")


