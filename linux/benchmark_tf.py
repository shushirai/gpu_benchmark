import tensorflow as tf
import argparse
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(description="TensorFlow CIFAR-10 Benchmark")
    parser.add_argument('--device', choices=['cpu', 'gpu', 'mps'], default=None,
                        help="Device to use: 'cpu', 'gpu', or 'mps'. Default: auto")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    return parser.parse_args()

def main():
    args = parse_args()

    # === デバイス設定 ===
    used_device = "cpu"
    if args.device == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

    elif args.device == 'gpu':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            used_device = "gpu"
        else:
            print("⚠️ GPU not available. Falling back to CPU.")

    elif args.device == 'mps':
        mps_devices = tf.config.list_physical_devices("GPU")
        if mps_devices:
            try:
                tf.config.experimental.set_visible_devices(mps_devices[0], "GPU")
                used_device = "mps"
            except Exception as e:
                print("⚠️ Failed to enable MPS:", e)
        else:
            print("⚠️ MPS (Metal) not available. Falling back to CPU.")

    else:
        # 自動判定
        if tf.config.list_physical_devices("GPU"):
            used_device = tf.test.gpu_device_name()

    print(f"Using device: {used_device}")

    # === データ準備 ===
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(args.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.batch_size)

    # === モデル構築 ===
    base_model = tf.keras.applications.ResNet50(
        weights=None, input_shape=(32, 32, 3), classes=10
    )
    base_model.compile(optimizer='sgd',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    # === 訓練 ===
    print("Starting training...")
    start_train = time.time()
    base_model.fit(train_ds,
                   epochs=args.epochs,
                   validation_data=test_ds,
                   verbose=1)
    print(f"Training time: {time.time() - start_train:.2f}s")

    # === 評価 ===
    print("Evaluating on test set...")
    start_eval = time.time()
    test_loss, test_acc = base_model.evaluate(test_ds, verbose=0)
    print(f"[{used_device}] Test Loss: {test_loss:.2f}, Accuracy: {test_acc * 100:.2f}%, Eval Time: {time.time() - start_eval:.2f}s")

if __name__ == '__main__':
    main()
