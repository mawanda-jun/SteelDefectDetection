import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tensorflow.python.keras.backend as K
dtype = 'float16'
K.set_floatx(dtype)
K.set_epsilon(1e-4)
# import numpy as np
import os, glob
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
from shutil import copy
from utils.logger import set_logger
from dataset_generator import DataGenerator
from Dataset.digest_train_csv import Digestive
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from segcaps import CapsNet, Container
from competition_losses_metrics import *
from config import conf


class SteelSeg:
    def __init__(self, conf):
        self.conf = conf

        self.data_reader = DataGenerator(conf, Digestive(conf).masks_at_least(1))

        self.model = self.build()

        # if reload_step is not zero it means we are reading data from already existing folders
        self.log_dir, self.model_dir, self.save_dir = self.set_dirs()

        # copy configuration inside log_dir
        copy(os.path.join(os.getcwd(), 'segcaps.py'), os.path.join(os.getcwd(), self.log_dir))
        copy(os.path.join(os.getcwd(), 'config.py'), os.path.join(os.getcwd(), self.log_dir))
        copy(os.path.join(os.getcwd(), 'main.py'), os.path.join(os.getcwd(), self.log_dir))
        copy(os.path.join(os.getcwd(), 'competition_losses_metrics.py'), os.path.join(os.getcwd(), self.log_dir))
        copy(os.path.join(os.getcwd(), 'dataset_generator.py'), os.path.join(os.getcwd(), self.log_dir))

        self.trained = False

    def build(self):
        # make placeholder to retrieve information about input shape
        w_res_bb = self.conf.img_w_res // self.conf.crops_w
        shape = (self.conf.img_h_res, w_res_bb, self.conf.dest_channels)

        inputs = tf.keras.Input(shape)
        # initialize model_on_input
        train_model = Container(shape=shape, n_class=self.conf.n_classes)
        train_model.build(inputs.shape)

        # print summary
        train_model.summary()
        return train_model

    def set_dirs(self):
        # set dirs per run
        model_dir = os.path.join(os.getcwd(), self.conf.results, self.conf.run_name, 'model_dir')
        log_dir = os.path.join(os.getcwd(), self.conf.results, self.conf.run_name, 'log_dir')
        save_dir = os.path.join(os.getcwd(), self.conf.results, self.conf.run_name, 'save_dir')
        # set experiment path
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        return log_dir, model_dir, save_dir

    def reload_weights(self):
        if self.conf.reload_step < 10:
            init_epoch = '0'+str(self.conf.reload_step)
        else:
            init_epoch = str(self.conf.reload_step)
        restore_filename_reg = 'weights.{}-*.hdf5'.format(init_epoch)
        restore_path_reg = os.path.join(self.model_dir, restore_filename_reg)
        list_files = glob.glob(restore_path_reg)
        assert len(list_files) > 0, 'ERR: No weights file match provided name {}'.format(
            restore_path_reg)

        # Take real filename
        restore_filename = list_files[0].split('/')[-1]
        restore_path = os.path.join(self.model_dir, restore_filename)

        assert os.path.isfile(restore_path), \
            'ERR: Weight file in path {} seems not to be a file'.format(restore_path)
        self.model.load_weights(restore_path)

    def setup_callables(self):
        monitor = "val_dice_coef"
        # Setup callback to save best weights after each epoch
        checkpointer = ModelCheckpoint(filepath=os.path.join(self.model_dir,
                                                             'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       monitor=monitor,
                                       mode='max')
        # setup callback to register training history
        csv_logger = CSVLogger(os.path.join(self.log_dir, 'log.csv'), append=True, separator=';')

        # setup logger to catch warnings and info messages
        set_logger(os.path.join(self.log_dir, 'train_val.log'))

        # setup callback to retrieve tensorboard info
        tensorboard = TensorBoard(log_dir=self.log_dir,
                                  write_graph=True,
                                  histogram_freq=0)

        # setup early stopping to stop training if val_loss is not increasing after 3 epochs
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=5,
            mode='max',
            verbose=0
        )
        lr_reducer = ReduceLROnPlateau(monitor=monitor, factor=0.05, cooldown=0, patience=5, verbose=0, mode='max')

        return [checkpointer, csv_logger, tensorboard, early_stopping, lr_reducer]

    def compile_model(self):
        # set optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.conf.learning_rate,
            beta_1=0.99,
            beta_2=0.999,
            epsilon=K.epsilon(),
        )
        # optimizer = tf.keras.optimizers.SGD(learning_rate=self.conf.learning_rate, nesterov=True, momentum=0.9)
        # optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")
        metrics = [dice_class_0, dice_class_1, dice_class_2, dice_class_3, dice_coef]
        # metrics = [dice_coef]
        self.model.compile(
            optimizer=optimizer,
            loss=generalised_dice_loss,
            metrics=metrics
        )

    def train(self):
        # if reload step is > 0 we would like to retrieve pre-trained weights from disk
        if self.conf.reload_step > 0:
            self.reload_weights()

        # set optimizer, loss and accuracy and compile model_on_input
        self.compile_model()

        # fit and validate model_on_input
        self.model.fit(
            self.data_reader.generate_train_set(),
            epochs=self.conf.epochs,
            validation_data=self.data_reader.generate_val_set(),
            steps_per_epoch=self.data_reader.steps_per_train_epoch,
            validation_steps=self.data_reader.steps_per_val_epoch,
            verbose=1,
            callbacks=self.setup_callables(),
            initial_epoch=self.conf.reload_step,
        )
        self.trained = True

    # def evaluate(self):
    #     if not self.trained:
    #         weight_to_be_restored = os.path.join(self.model_dir, self.conf.eval_weight)
    #         if not os.path.isfile(weight_to_be_restored):
    #             raise FileNotFoundError('Weight not found. Please double check trial_dir, run_name and eval_weight')
    #         self.model.load_weights(weight_to_be_restored, by_name=True)
    #         self.compile_model('test')
    #     results = self.model.evaluate(
    #         self.data_reader.generate_test_set(),
    #         verbose=1,
    #         steps=self.data_reader.num_test_batch,
    #     )
    #     with open(os.path.join(self.log_dir, 'test_{}.csv'.format(self.conf.eval_weight)), 'w') as f:
    #         f.write("test loss, test acc: {}".format(results))


if __name__ == '__main__':
    # with tf.device('/gpu:0'):
    steelseg = SteelSeg(conf)
    steelseg.train()



