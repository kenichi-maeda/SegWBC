import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input, Dropout
from tensorflow.keras.models import Model
import cv2
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.losses import binary_crossentropy

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and will be used for training.")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("No GPU found. Training will be done on CPU.")

# Define UNet model
def unet_model(input_size=(128, 128, 1)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    d = Dropout(0.5)(c4)

    # Decoder
    u5 = UpSampling2D((2, 2))(d)
    m5 = Concatenate()([u5, c3])
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(m5)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    m6 = Concatenate()([u6, c2])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(m6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    m7 = Concatenate()([u7, c1])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(m7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs, outputs)
    return model

# Load and preprocess data
def load_data(image_dir, image_size=(128, 128), limit=None):
    X, Y = [], []
    count = 0
    for file in sorted(os.listdir(image_dir)):
        if file.endswith("_HE.png"):
            if limit is not None and count >= limit:
                break

            img = cv2.imread(os.path.join(image_dir, file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)
            X.append(img)

            mask_file = file.replace("_HE.png", "_mask.png")
            mask = cv2.imread(os.path.join(image_dir, mask_file), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, image_size)
            mask = (mask > 0).astype("float32")
            Y.append(mask)

            count += 1

    X = np.expand_dims(np.array(X).astype("float32") / 255.0, axis=-1)
    Y = np.expand_dims(np.array(Y).astype("float32"), axis=-1)
    return X, Y

# Perform data augmentation
def get_augmented_dataset(X, Y, batch_size=2, seed=42):
    import tensorflow as tf
    tf.random.set_seed(seed)
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))

    def augment_fn(x, y):
        # Flip images
        x = tf.image.random_flip_left_right(x, seed=seed)
        y = tf.image.random_flip_left_right(y, seed=seed)

        # Rotate images
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32, seed=seed)
        x = tf.image.rot90(x, k=k)
        y = tf.image.rot90(y, k=k)

        # Change brightness
        x = tf.image.random_brightness(x, max_delta=0.1, seed=seed)

        return x, y

    return dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)\
                  .shuffle(buffer_size=len(X), seed=seed)\
                  .batch(batch_size)\
                  .prefetch(tf.data.AUTOTUNE)

# Compute dice coefficient
def dice_coefficient(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-7)

# Compute Jaccard index (IoU)
def jaccard_index(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return intersection / (union + 1e-7)


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (numerator + 1e-7) / (denominator + 1e-7)

def combo_loss(y_true, y_pred):
    d = dice_loss(y_true, y_pred)
    b = binary_crossentropy(y_true, y_pred)
    return 0.5 * d + 0.5 * b

# Count number of blobs in a image
def count_blobs(mask):
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask_uint8 = (mask * 255).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(mask_uint8)
    return num_labels - 1  # subtract background label (0)

# Compute counting accuracy
def counting_accuracy(y_true, y_pred):
    n_true = count_blobs(y_true)
    n_pred = count_blobs(y_pred)
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    return 1 - abs(n_pred - n_true) / n_true

# Main function
def main():
    image_dir = "./CD45RB_Leukocyte"
    X, Y = load_data(image_dir, limit=8000)
    print("Loaded data:", X.shape, Y.shape)

    output_dir = "AS_lUNet"
    os.makedirs(output_dir, exist_ok=True)

    # Split into train/val/test
    train_X, temp_X, train_Y, temp_Y = train_test_split(X, Y, test_size=0.3, random_state=42)
    val_X, test_X, val_Y, test_Y = train_test_split(temp_X, temp_Y, test_size=0.5, random_state=42)

    model = unet_model()
    model.compile(optimizer="adam", loss=combo_loss, metrics=["accuracy"])
    model.summary()

    train_dataset = get_augmented_dataset(train_X, train_Y, batch_size=2)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_Y)).batch(2)

    start = time.time()
    history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)
    duration = time.time() - start
    print(f"Training time: {duration:.2f} seconds")

    # Save accuracy/loss plots
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'lUNet_accuracy_per_epoch.png'))
    plt.clf()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'lUNet_loss_per_epoch.png'))
    plt.clf()

    # Save model
    model.save(os.path.join(output_dir, "unet_model.h5"))

    # Evaluation on test set
    #test_preds = model.predict(test_X, batch_size=4)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_X).batch(4)
    test_preds = model.predict(test_dataset)

    test_preds_binary = (test_preds > 0.5).astype(np.uint8)
    test_Y_binary = test_Y.astype(np.uint8)

    dice = dice_coefficient(test_Y_binary, test_preds_binary)
    jaccard = jaccard_index(test_Y_binary, test_preds_binary)
    print(f"Dice Coefficient: {dice:.4f}")
    print(f"Jaccard Index (IoU): {jaccard:.4f}")

    # Flatten for confusion matrix
    y_true_flat = test_Y_binary.flatten()
    y_pred_flat = test_preds_binary.flatten()
    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat, labels=[0,1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)

    blob_accuracies = []
    for i in range(len(test_X)):
        blob_acc = counting_accuracy(test_Y_binary[i], test_preds_binary[i])
        blob_accuracies.append(blob_acc)
    avg_blob_accuracy = np.mean(blob_accuracies)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Average Blob Counting Accuracy: {avg_blob_accuracy:.4f}")

    # Save evaluation metrics to CSV
    with open(os.path.join(output_dir, "metrics.csv"), "w") as f:
        f.write("Metric,Value\n")
        f.write(f"Dice,{dice:.4f}\n")
        f.write(f"IoU,{jaccard:.4f}\n")
        f.write(f"Accuracy,{accuracy:.4f}\n")
        f.write(f"Sensitivity,{sensitivity:.4f}\n")
        f.write(f"Specificity,{specificity:.4f}\n")
        f.write(f"TrainingTimeSeconds,{duration:.2f}\n")
        f.write(f"BlobCountingAccuracy,{avg_blob_accuracy:.4f}\n")

    # Save a few sample predictions
    pred_dir = os.path.join(output_dir, "sample_predictions")
    os.makedirs(pred_dir, exist_ok=True)
    for i in range(5):
        original = (test_X[i].squeeze() * 255).astype(np.uint8)
        gt_mask = (test_Y[i].squeeze() * 255).astype(np.uint8)
        pred_mask = (test_preds_binary[i].squeeze() * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(pred_dir, f"image_{i}_original.png"), original)
        cv2.imwrite(os.path.join(pred_dir, f"image_{i}_gt_mask.png"), gt_mask)
        cv2.imwrite(os.path.join(pred_dir, f"image_{i}_pred_mask.png"), pred_mask)


if __name__ == "__main__":
    main()