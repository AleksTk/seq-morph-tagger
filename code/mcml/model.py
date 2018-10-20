import numpy as np
import tensorflow as tf
from collections import OrderedDict

from .data_utils import pad_sequences, labels2matrix
from base_model import BaseModel


class Model(BaseModel):
    """Mcml model"""

    def __init__(self, config):
        super(Model, self).__init__(config)
        assert isinstance(self.config.vocab_tags, OrderedDict)
        self.idx_to_tag = OrderedDict()
        for cat, cat_dict in self.config.vocab_tags.items():
            self.idx_to_tag[cat] = {idx: tag for tag, idx in cat_dict.items()}

    def add_placeholders(self):
        """Define placeholders for computational graph"""
        super().add_placeholders()

        for category in self.idx_to_tag:
            labels = tf.placeholder(tf.int32,
                                    shape=[None, None],
                                    name="labels_{}".format(self.escape_category(category)))
            self.setattr("labels", labels, category)

    def add_decoder_op(self):
        output_h = self.encoder_output[0]
        nsteps = tf.shape(output_h)[1]
        output = tf.reshape(output_h, [-1, 2 * self.config.hidden_size_lstm])

        # get POS logits
        category = "POS"
        category_size = len(self.idx_to_tag[category])
        pos_pred = tf.layers.dense(output, category_size,
                                   name="Output_{}".format(self.escape_category(category)))
        pos_logits = tf.reshape(pos_pred, [-1, nsteps, category_size])
        self.setattr("logits", pos_logits, category)

        if self.config.share_pos_logits:
            pos_pred = tf.layers.dropout(pos_pred, rate=0.5, training=self.training_phase)
            output = tf.concat([output, pos_pred], axis=-1)

        # get logits for other categories
        for category in self.idx_to_tag:
            if category != "POS":
                category_size = len(self.idx_to_tag[category])
                pred = tf.layers.dense(output, category_size,
                                       name="Output_{}".format(self.escape_category(category)))
                logits = tf.reshape(pred, [-1, nsteps, category_size])
                self.setattr("logits", logits, category)

    def add_pred_op(self):
        for category in self.idx_to_tag:
            logits = self.getattr("logits", category)
            labels_pred = tf.argmax(logits, -1)
            self.setattr("labels_pred", labels_pred, category)

    def add_loss_op(self):
        """Defines the loss"""
        loss_dict = {}
        for category in self.idx_to_tag:
            logits = self.getattr("logits", category)
            labels = self.getattr("labels", category)

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            loss = tf.reduce_mean(losses)
            loss_dict[category] = loss

        if self.config.loss_strategy == 'sum':
            loss = tf.reduce_sum(list(loss_dict.values()))
        elif self.config.loss_strategy == 'balanced':
            pos_loss = loss_dict.pop("POS")
            loss = pos_loss * len(loss_dict) + tf.reduce_sum(list(loss_dict.values()))
        elif self.config.loss_strategy == 'trainable_linear_combination':
            g = tf.get_variable(name="loss_linear_comb_parameter", shape=[1])
            alpha = tf.sigmoid(g)
            pos_loss = loss_dict.pop("POS")
            loss = alpha * pos_loss + (1 - alpha) * tf.reduce_sum(list(loss_dict.values()))
        else:
            raise ValueError("Invalid loss_strategy %s" % self.config.loss_strategy)
        self.loss = loss

    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_encoder_op()
        self.add_decoder_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op(self.config.lr_method, self.lr, self.loss, self.config.clip)
        self.initialize_session()

    def predict_batch(self, words, category):
        fd = self.prebuild_feed_dict_batch(words)
        fd, sequence_lengths = self.get_final_feed_dict(False, fd)
        labels_pred = self.getattr("labels_pred", category)
        labels_pred = self.sess.run([labels_pred], feed_dict=fd)[0]
        return labels_pred, sequence_lengths

    def predict_batch_with_logits(self, words, category):
        """
        Args:
            words: list of sentences
            category: category id

        Returns:
            labels_pred: list of labels for each sentence for the given category
            sequence_length

        """
        fd = self.prebuild_feed_dict_batch(words)
        fd, sequence_lengths = self.get_final_feed_dict(False, fd)

        logits = self.getattr("logits", category)
        labels_pred = self.getattr("labels_pred", category)
        labels_pred, labels_logits = self.sess.run([labels_pred, logits], feed_dict=fd)
        return labels_pred, labels_logits, sequence_lengths

    def escape_category(self, category):
        return category.replace('[', '_').replace(']', '_')

    def getattr(self, name, category):
        return getattr(self, "{}_{}".format(name, self.escape_category(category)))

    def setattr(self, name, value, category):
        setattr(self, "{}_{}".format(name, self.escape_category(category)), value)

    def evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            accuracy: (float)

        """
        pos = 0.
        total = 0
        batch_size = self.config.eval_batch_size

        for feed_dict in self.iter_prebuilt_feed_dict_batches(test, batch_size):
            labels_true_dict = feed_dict.pop('labels')
            fd, sequence_lengths = self.get_final_feed_dict(False, feed_dict)
            preds = np.ones((len(sequence_lengths), np.max(sequence_lengths)), dtype=np.bool)
            for category_idx, category in enumerate(self.idx_to_tag):
                labels_true = labels_true_dict[category]
                labels_pred = self.getattr("labels_pred", category)
                labels_pred = self.sess.run([labels_pred], feed_dict=fd)[0]
                p = labels_pred == labels_true
                preds = np.bitwise_and(preds, p)
            mask = self.sess.run(tf.sequence_mask(sequence_lengths))
            masked_preds = preds[mask]
            pos += np.sum(masked_preds)
            total += masked_preds.shape[0]
        acc = pos / total
        return acc

    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word_infer(w) for w in words_raw]
        if type(words[0]) == tuple:
            # words = sentence = tuple of (char-ids, word-ids)
            words = list(zip(*words))

        preds = [[] for _ in range(len(words_raw))]
        for category in self.idx_to_tag:
            batch_pred = self.predict_batch([words], category)[0]
            sentence_pred = batch_pred[0]
            for i, word_pred in enumerate(sentence_pred):
                if word_pred != 0:
                    tag = self.idx_to_tag[category][word_pred]
                    preds[i].append(tag)
        return preds

    def predict_logits(self, words_raw):
        """Returns a list of raw logits for all tags in alphabet.

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of logist (string), one for each word in the sentence

        """
        words = [self.config.processing_word_infer(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = list(zip(*words))
        pred_ids, pred_logits, _ = self.predict_batch([words])
        sentence_logits = pred_logits[0]
        return sentence_logits

    def get_final_feed_dict(self, training_phase, feed_dict, lr=None):
        feed = {
            self.training_phase: training_phase,
            self.word_ids: feed_dict['word_ids'],
            self.sequence_lengths: feed_dict['sequence_lengths']
        }

        if self.config.use_char_embeddings:
            feed[self.char_ids] = feed_dict['char_ids']
            feed[self.word_lengths] = feed_dict['word_lengths']

        if 'labels' in feed_dict:
            for category_idx, category in enumerate(self.idx_to_tag):
                label_matrix = feed_dict['labels'][category]
                feed[self.getattr("labels", category)] = label_matrix

        if lr is not None:
            feed[self.lr] = lr

        sequence_lengths = feed_dict['sequence_lengths']
        return feed, sequence_lengths

    def prebuild_feed_dict_batch(self, words, labels=None):
        if self.config.use_char_embeddings:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        feed = {
            'word_ids': word_ids,
            'sequence_lengths': sequence_lengths
        }

        if self.config.use_char_embeddings:
            feed['char_ids'] = char_ids
            feed['word_lengths'] = word_lengths

        if labels is not None:
            max_sentence_length = max(len(snt) for snt in labels)
            cat2labels_dict = {}
            for category_idx, category in enumerate(self.idx_to_tag):
                label_matrix = labels2matrix(labels, category_idx, max_sentence_length)
                cat2labels_dict[category] = label_matrix
            feed['labels'] = cat2labels_dict
        return feed
