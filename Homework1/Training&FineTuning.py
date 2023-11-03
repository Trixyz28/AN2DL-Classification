import tensorflow as tf
import tensorflow.keras as tfk
from albumentations import *
import albumentations as alb
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from keras import backend as K
from datetime import datetime
import math
from keras.callbacks import Callback
from keras import backend as K
import os

tfk = tf.keras
tfkl = tf.keras.layers
seed = 23
input_shape = (224, 224, 3)
batch_size = 32


class CosineAnnealingScheduler(Callback):
    """
    Define the Cosine annealing scheduler.
    Reference: https://github.com/4uiiurz1/keras-cosine-annealing/blob/master/cosine_annealing.py
    """

    def __init__(self, T_max, eta_max, eta_min=0., verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


def categorical_focal_loss(y_true, y_pred):
    """
    Define the Categorical focal loss.
    """
    alpha = np.array([0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15], dtype=np.float32)
    gamma = 2
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

    # Compute mean loss in mini_batch
    return K.mean(K.sum(loss, axis=-1))


def main(model, i, j, scheduler=False, epoch=80):
    def create_folders_and_callbacks(model_name):
        exps_dir = os.path.join('experiments')
        if not os.path.exists(exps_dir):
            os.makedirs(exps_dir)

        now = datetime.now().strftime('%b%d_%H-%M-%S')
        print(now)

        exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        callbacks = []

        # Model checkpoint
        ckpt_dir = os.path.join(exp_dir, 'base_ckpts')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'),
                                                           save_weights_only=False,  # True to save only weights
                                                           save_best_only=True,  # True to save only the best epoch
                                                           monitor='val_accuracy')
        callbacks.append(ckpt_callback)

        # Visualize Learning on Tensorboard
        tb_dir = os.path.join(exp_dir, 'tb_logs')
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)

        # By default shows losses and metrics for both training and validation
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                                     profile_batch=0,
                                                     histogram_freq=1)  # if > 0 (epochs) shows weights histograms
        callbacks.append(tb_callback)

        return callbacks, now


    class AugmentDataGenerator(Sequence):
        """
        Define the generator of augmented data.
        """
        def __init__(self, datagen, augment=None):
            self.datagen = datagen
            if augment is None:
                self.augment = alb.Compose([])
            else:
                self.augment = augment

            self.image = []

        def __len__(self):
            return len(self.datagen)

        def __getitem__(self, x):
            images, *rest = self.datagen[x]
            augmented = []
            for image in images:
                image = self.augment(image=image)['image']
                augmented.append(image)

            return (np.array(augmented), *rest)

    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

    # Set batch size
    train_data_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input)
    val_data_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input)
    print("Rescaling completed")

    # Create iterators
    train_gen = train_data_gen.flow_from_directory(
        directory='KF_data/train_' + str(i) + '/',
        target_size=(96, 96),
        color_mode='rgb',
        classes=None,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        seed=23
    )

    val_gen = val_data_gen.flow_from_directory(
        directory='KF_data/val_' + str(i) + '/',
        target_size=(96, 96),
        color_mode='rgb',
        classes=None,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False,
        seed=23
    )

    # Apply data augmentation methods
    train_gen_augment = AugmentDataGenerator(train_gen, alb.Compose([
        alb.Resize(224, 224),
        alb.HorizontalFlip(),
        alb.VerticalFlip(),
        alb.Transpose(),
        alb.RandomRotate90(),

        alb.RandomBrightnessContrast(),
        alb.ShiftScaleRotate(),

        alb.OneOf([
            alb.OpticalDistortion(p=1),
            alb.GridDistortion(p=1),
            alb.ElasticTransform(p=1)
        ], p=0.3),

        alb.GaussNoise(),
        alb.CoarseDropout(max_height=32, max_width=32, max_holes=16),
    ]))

    val_gen_augment = AugmentDataGenerator(val_gen, alb.Compose([
        alb.Resize(224, 224)]))
    # ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

    if scheduler:
        callbacks, now = create_folders_and_callbacks("Final_" + str(i) + "_" + str(j))
        callbacks.append(CosineAnnealingScheduler(T_max=40, eta_max=1e-1, eta_min=1e-3))
        optimizer = tfk.optimizers.SGD(learning_rate=1e-1)
        loss = categorical_focal_loss
    else:
        callbacks, now = create_folders_and_callbacks("Final_" + str(i))
        optimizer = tfk.optimizers.Adam(learning_rate=1e-5)
        loss = tfk.losses.CategoricalCrossentropy()

    metrics = ['accuracy']

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    model.summary()

    epochs = epoch
    history = model.fit(
        train_gen_augment,
        epochs=epochs,
        validation_data=val_gen_augment,
        callbacks=callbacks
    ).history

    return model


if __name__ == "__main__":

    # Train K-Fold models
    
    for i in range(5):
        supernet = tfk.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )

        model = tf.keras.Sequential()
        model.add(supernet)
        model.add(tfkl.GlobalAvgPool2D())
        model.add(tfkl.Dropout(0.3))
        model.add(tfkl.Dense(
            units=8,
            activation='softmax',
            kernel_initializer=tfk.initializers.GlorotUniform(seed)))
        main(model, i=i, j=0, scheduler=False, epoch=1000)


    # Fine-tuning Model
    T = 3
    model = tfk.models.load_model(
        "experiments/FT_KF_EFF_B0_0Cos_1_Nov23_07-25-32" + "/base_ckpts/cp_" + str('68') + ".ckpt",
        custom_objects={"categorical_focal_loss": categorical_focal_loss})
    for k in range(T):
        model = main(model, 0, k)

    model = tfk.models.load_model(
        "experiments/FT_KF_EFF_B0_1Cos_1_Nov23_08-23-25" + "/base_ckpts/cp_" + str('26') + ".ckpt",
        custom_objects={"categorical_focal_loss": categorical_focal_loss})
    for k in range(T):
        model = main(model, 1, k)

    model = tfk.models.load_model(
        "experiments/FT_KF_EFF_B0_2Cos_1_Nov23_09-26-08" + "/base_ckpts/cp_" + str('28') + ".ckpt",
        custom_objects={"categorical_focal_loss": categorical_focal_loss})
    for k in range(T):
        model = main(model, 2, k)

    model = tfk.models.load_model(
        "experiments/FT_KF_EFF_B0_3Cos_1_Nov23_10-26-20" + "/base_ckpts/cp_" + str('33') + ".ckpt",
        custom_objects={"categorical_focal_loss": categorical_focal_loss})
    for k in range(T):
        model = main(model, 3, k)

    model = tfk.models.load_model(
        "experiments/FT_KF_EFF_B0_4Cos_1_Nov23_11-30-17" + "/base_ckpts/cp_" + str('47') + ".ckpt",
        custom_objects={"categorical_focal_loss": categorical_focal_loss})
    for k in range(T):
        model = main(model, 4, k)
