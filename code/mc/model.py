import numpy as np
import tensorflow as tf

from .data_utils import minibatches, pad_sequences
from base_model import BaseModel


class Model(BaseModel):
    """Multi-class (MC) model"""

    def __init__(self, config):
        super(Model, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}

    def add_placeholders(self):
        super().add_placeholders()
        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

    def add_decoder_op(self):
        output_h = self.encoder_output[0]
        nsteps = tf.shape(output_h)[1]
        output = tf.reshape(output_h, [-1, 2 * self.config.hidden_size_lstm])
        pred = tf.layers.dense(output, self.config.ntags)
        self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

    def add_pred_op(self):
        """Defines self.labels_pred"""
        self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def add_loss_op(self):
        """Defines the loss"""
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.labels)
        mask = tf.sequence_mask(self.sequence_lengths)
        losses = tf.boolean_mask(losses, mask)
        self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_encoder_op()
        self.add_decoder_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op(self.config.lr_method, self.lr, self.loss, self.config.clip)
        self.initialize_session()

    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd = self.prebuild_feed_dict_batch(words)
        fd, sequence_lengths = self.get_final_feed_dict(False, fd)
        labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)
        return labels_pred, sequence_lengths

    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word_infer(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds

    def evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            accuracy: (float)

        """
        accs = []
        for words, labels in minibatches(test, self.config.eval_batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)
            for lbls_true, lbls_pred, length in zip(labels, labels_pred, sequence_lengths):
                # compare sentence labels
                lbls_true = lbls_true[:length]
                lbls_pred = lbls_pred[:length]
                accs += [a == b for (a, b) in zip(lbls_true, lbls_pred)]
        acc = np.mean(accs)
        return acc

    def prebuild_feed_dict_batch(self, words, labels=None):
        # perform padding of the given data
        if self.config.use_char_embeddings:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {'word_ids': word_ids,
                'sequence_lengths': sequence_lengths}

        if self.config.use_char_embeddings:
            feed['char_ids'] = char_ids
            feed['word_lengths'] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed['labels'] = labels

        return feed

    def get_final_feed_dict(self, training_phase, feed_dict, lr=None):
        fd = {self.training_phase: training_phase,
              self.word_ids: feed_dict['word_ids'],
              self.sequence_lengths: feed_dict['sequence_lengths']
              }

        if self.config.use_char_embeddings:
            fd[self.char_ids] = feed_dict['char_ids']
            fd[self.word_lengths] = feed_dict['word_lengths']

        if 'labels' in feed_dict:
            fd[self.labels] = feed_dict['labels']

        if lr is not None:
            fd[self.lr] = lr

        return fd, feed_dict['sequence_lengths']
