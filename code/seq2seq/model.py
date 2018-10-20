import numpy as np
import tensorflow as tf

from .data_utils import minibatches, pad_sequences, PAD, SOS, EOS
from base_model import BaseModel


class Model(BaseModel):
    """Sequence model"""

    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.sess = None
        self.saver = None

        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}
        self.idx_to_word = {idx: word for word, idx in
                            self.config.vocab_words.items()}
        self.idx_to_char = {idx: c for c, idx in
                            self.config.vocab_chars.items()}

        self.sos_id = self.config.vocab_tags[SOS]
        self.eos_id = self.config.vocab_tags[EOS]
        self.pad_id = self.config.vocab_tags[PAD]
        assert self.pad_id == 0

    def add_placeholders(self):
        """Define placeholders for computational graph"""

        self.training_phase = tf.placeholder(tf.bool, shape=[], name="training_phase")
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.tag_ids = tf.placeholder(tf.int32, shape=[None, None], name="tag_ids")
        self.tag_lengths = tf.placeholder(tf.int32, shape=[None], name="tag_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")

        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def add_tag_embeddings_op(self):
        tag_embeddings = tf.get_variable(
            name="tag_embeddings",
            dtype=tf.float32,
            shape=[self.config.ntags, self.config.dim_tag])
        self.tag_embeddings = tag_embeddings

    def add_decoder_op(self):
        # reshape inputs to a list of words
        input_mask = tf.sequence_mask(self.sequence_lengths)
        encoder_output_h, encoder_output_c = self.encoder_output
        decoder_input_h = tf.boolean_mask(encoder_output_h, input_mask)
        decoder_input_c = tf.boolean_mask(encoder_output_c, input_mask)
        initial_state = tf.contrib.rnn.LSTMStateTuple(h=decoder_input_h, c=decoder_input_c)

        batch_size = tf.shape(decoder_input_h)[0]
        projection_layer = tf.layers.Dense(self.config.ntags, use_bias=True, name="decoder_proj")

        decoder_cell = tf.contrib.rnn.LSTMCell(num_units=2 * self.config.hidden_size_lstm)

        start_tokens = tf.tile([self.sos_id], [batch_size])

        # shift tags one step to the left and prepend 'sos' token.
        tag_ids_train = tf.concat([tf.expand_dims(start_tokens, 1), self.tag_ids[:, :-1]], 1)
        tags_train_embedded = tf.nn.embedding_lookup(self.tag_embeddings, tag_ids_train)
        tags_train_embedded = tf.layers.dropout(tags_train_embedded,
                                                rate=1 - self.config.tag_embeddings_dropout,
                                                training=self.training_phase)

        # Training
        train_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=tags_train_embedded,
            sequence_length=self.tag_lengths  # `tag-length` covers <sos-token, actual tags, eos-token>
        )

        train_decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell,
            train_helper,
            initial_state=initial_state,
            output_layer=projection_layer)

        decoder_outputs, final_state, decoder_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(train_decoder)

        logits = decoder_outputs.rnn_output
        logits = tf.verify_tensor_all_finite(logits, "Logits not finite")

        # from padded training tags extracts actual-tags + eos-token:
        weights = tf.to_float(tf.not_equal(tag_ids_train, self.eos_id))
        weights = tf.to_float(tf.not_equal(weights, self.pad_id))
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                targets=self.tag_ids,
                                                weights=weights,
                                                name="sequence_loss",
                                                average_across_timesteps=False)
        self.loss = tf.reduce_sum(loss)

        # Inference
        infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=self.tag_embeddings,
            start_tokens=start_tokens,
            end_token=self.eos_id)

        infer_decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell,
            infer_helper,
            initial_state=initial_state,
            output_layer=projection_layer)

        final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            infer_decoder,
            maximum_iterations=self.config.decoder_maximum_iterations,
            impute_finished=True)

        decoder_logits = final_outputs.rnn_output
        decoder_logits = tf.verify_tensor_all_finite(decoder_logits, "Decoder Logits not finite")
        with tf.control_dependencies([tf.assert_rank(decoder_logits, 3),
                                      tf.assert_none_equal(tf.reduce_sum(decoder_logits), 0.),
                                      tf.assert_equal(tf.cast(tf.argmax(decoder_logits, axis=-1), tf.int32),
                                                      final_outputs.sample_id)]):
            decoder_logits = tf.identity(decoder_logits)

        self.decoder_logits = decoder_logits
        self.labels_pred = final_outputs.sample_id
        self.labels_pred_lengths = final_sequence_lengths

    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_tag_embeddings_op()
        self.add_encoder_op()
        self.add_decoder_op()
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
        labels_pred, labels_pred_lengths = self.sess.run([self.labels_pred, self.labels_pred_lengths], feed_dict=fd)
        return labels_pred, labels_pred_lengths

    def predict(self, words_raw):
        """Returns a list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word_infer(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = list(zip(*words))
        labels_pred, labels_pred_lengths = self.predict_batch([words])
        preds = [[self.idx_to_tag[tag_id] for tag_id in pred[:length][:-1]]
                 for pred, length in zip(labels_pred, labels_pred_lengths)]
        return preds

    def evaluate(self, test):
        """Evaluates performance on test set"""
        accs = []
        for words, labels in minibatches(test, self.config.eval_batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)
            labels_true = [label for sentence in labels for label in sentence]
            for lbl_true, lbl_pred, length in zip(labels_true, labels_pred, sequence_lengths):
                lbl_pred = list(lbl_pred[:length])[:-1]
                is_correct = int(lbl_true == lbl_pred)
                accs.append(is_correct)
        acc = np.mean(accs)
        return acc

    def prebuild_feed_dict_batch(self, words, labels=None):
        """Given a batch of words, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of words. A word is a list of ids
            labels: list of ids

        Returns:
            dict {placeholder: value}

        """
        feed = {}

        # padding
        if self.config.use_char_embeddings is True and self.config.use_word_embeddings is True:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, self.pad_id)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=self.pad_id, nlevels=2)
        elif self.config.use_word_embeddings is True:
            word_ids, sequence_lengths = pad_sequences(words, self.pad_id)
        elif self.config.use_char_embeddings is True:
            char_ids, word_lengths = pad_sequences(words, pad_tok=self.pad_id, nlevels=2)
            sequence_lengths = [len(snt) for snt in words]

        if self.config.use_word_embeddings:
            feed['word_ids'] = word_ids

        if self.config.use_char_embeddings:
            feed['char_ids'] = char_ids
            feed['word_lengths'] = word_lengths

        feed['sequence_lengths'] = sequence_lengths

        if labels is not None:
            word_tags = [word_tags + [self.eos_id, self.pad_id]
                         for sentence in labels for word_tags in sentence]
            tag_ids, tag_lengths = pad_sequences(word_tags, self.pad_id)
            feed['tag_ids'] = tag_ids
            feed['tag_lengths'] = tag_lengths  # Word tags include actual tags + eos-token + 1 pad-token.

        return feed

    def get_final_feed_dict(self, training_phase, feed_dict, lr=None):
        fd = {self.training_phase: training_phase,
              self.sequence_lengths: feed_dict['sequence_lengths']}

        if self.config.use_word_embeddings:
            fd[self.word_ids] = feed_dict['word_ids']

        if self.config.use_char_embeddings:
            fd[self.char_ids] = feed_dict['char_ids']
            fd[self.word_lengths] = feed_dict['word_lengths']

        if 'tag_ids' in feed_dict:
            fd[self.tag_ids] = feed_dict['tag_ids']
            fd[self.tag_lengths] = feed_dict['tag_lengths']

        if lr is not None:
            fd[self.lr] = lr

        sequence_lengths = feed_dict['sequence_lengths']
        return fd, sequence_lengths
