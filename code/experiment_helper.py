import os
from collections import defaultdict
from itertools import product
import importlib

import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as plt

plt.style.use('ggplot')
import tensorflow as tf
import numpy as np
import pandas as pd

from utils import print_config, load_config_from_file
import evaluation

import common_config


def evaluate(model, config_holder, test_file, lang_key, eval_type, out_dir):
    df = evaluation.predict(model, config_holder, test_file)
    acc_dict = evaluation.calculate_accuracy(df)
    acc_verbose = evaluation.accuracy_to_string_verbose(acc_dict)
    evaluation.save_results(df, acc_dict, acc_verbose, lang_key, eval_type, out_dir)


def _collect_results(memory, iter_result):
    memory.append(iter_result["dev_acc"])


def _update_output_dir(config, out_dir):
    config.out_dir = out_dir
    config.out_data_dir = os.path.join(out_dir, "data")
    config.dir_output = os.path.join(out_dir, "results")
    config.dir_model = os.path.join(config.dir_output, "model.weights")
    config.path_log = os.path.join(config.dir_output, "log.txt")


def train_model(model_module, config, params=None, values=None, do_evaluate=False):
    if params is not None:
        for param, value in zip(params, values):
            assert param in config.__dict__
            setattr(config, param, value)
    print_config(config)

    data_builder = model_module.DataBuilder(config)
    data_builder.run()

    config_holder = model_module.ConfigHolder(config)
    model = model_module.Model(config_holder)
    model.build()

    train = model_module.CoNLLDataset(config_holder.filename_train,
                                      config_holder.processing_word_train,
                                      config_holder.processing_tag,
                                      config_holder.max_iter,
                                      use_buckets=config.bucket_train_data,
                                      batch_size=config.batch_size,
                                      shuffle=config.shuffle_train_data,
                                      sort=config.sort_train_data)
    test = model_module.CoNLLDataset(config_holder.filename_dev,
                                     config_holder.processing_word_infer,
                                     config_holder.processing_tag,
                                     config_holder.max_iter,
                                     sort=True)
    train_eval = model_module.CoNLLDataset(config_holder.filename_train,
                                           config_holder.processing_word_infer,
                                           config_holder.processing_tag,
                                           sort=True,
                                           max_iter=config_holder.train_sentences_to_eval)
    model.train(train, test, train_eval)
    model.close_session()
    tf.reset_default_graph()

    # read accuracies for all iterations
    df = pd.read_csv(config_holder.training_log,
                     names=['epoch', 'acc_train', 'acc_test', 'train_loss', 'nbatches',
                            'epoch_time', 'train_time', 'eval_time'])
    acc_train_list = df['acc_train']
    acc_test_list = df['acc_test']
    train_loss_list = df['train_loss']

    # evaluate
    if do_evaluate is True:
        print("Evaluating...")
        evaluate(model, config_holder, config_holder.filename_dev, 'LANG_NA', 'dev', config_holder.out_dir)
        model.close_session()
        tf.reset_default_graph()

    return acc_train_list, acc_test_list, train_loss_list


def plot_experiment(acc_train_list, acc_test_list, train_loss_list, image_file_acc, image_file_loss, title):
    # plot test accuracy
    plt.plot(range(len(acc_test_list)), acc_test_list, '-', c="red",
             label="TEST : (best=%.4f, iter %d)" % (max(acc_test_list), np.argmax(acc_test_list)))
    # plot best test accuracy as vertical lines
    plt.axvline(x=np.argmax(acc_test_list), linestyle='--', c="red", label='best test acc')

    # plot train accuracy
    plt.plot(range(len(acc_train_list)), acc_train_list, '-', c="blue",
             label="TRAIN: (best=%.4f, iter %d)" % (max(acc_train_list), np.argmax(acc_train_list)))
    # plot best test accuracy as vertical lines
    plt.axvline(x=np.argmax(acc_train_list), linestyle='--', c="blue", label='best train acc')

    # plot train/test accuracy
    plt.title(title)
    plt.xlabel("Epoch")
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.ylabel("accuracy")
    plt.legend(loc=4)
    plt.grid(color='white')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(image_file_acc, dpi=100)
    plt.clf()
    plt.cla()
    plt.close()

    # plot loss
    plt.plot(range(len(train_loss_list)), train_loss_list, '-',
             label="best=%.4f, iter %d" % (np.min(train_loss_list), np.argmin(train_loss_list)))
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("train loss")
    plt.legend(loc=3)
    plt.grid(color='white')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(image_file_loss, dpi=100)
    plt.clf()
    plt.cla()
    plt.close()


def run_experiment(model_module, config, title=None):
    acc_train_list, acc_test_list, train_loss_list = train_model(model_module, config)

    print("Best test acc %f (epoch %d)" % (max(acc_test_list), np.argmax(acc_test_list)))
    print("Iteration test acc:", list(acc_test_list))

    plot_experiment(acc_train_list, acc_test_list, train_loss_list,
                    image_file_acc=os.path.join(config.out_dir, 'accuracy.png'),
                    image_file_loss=os.path.join(config.out_dir, "train_loss.png"),
                    title=title if title else "Experiment accuracy")
