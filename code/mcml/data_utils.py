from common_data_utils import *
from collections import OrderedDict, defaultdict
import pickle

UNK = "__UNK__"
NUM = "__NUM__"
NONE = "__O__"

SOS = "__SOS__"
EOS = "__EOS__"
PAD = "__PAD__"


class ConfigHolder(BaseConfigHolder):
    def __init__(self, config):
        for k, v in config.__dict__.items():
            if not k.startswith('__') and not callable(v):
                setattr(self, k, v)

        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags = load_tag_dict(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)
        self.vocab_singletons = load_vocab(self.filename_singletons) if self.train_singletons else None

        self.nwords = len(self.vocab_words)
        self.nchars = len(self.vocab_chars)
        self.ntags = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word_train = get_processing_word(self.vocab_words,
                                                         self.vocab_chars,
                                                         vocab_singletons=self.vocab_singletons,
                                                         singleton_p=self.singleton_p,
                                                         lowercase=self.lowercase,
                                                         use_words=self.use_word_embeddings,
                                                         use_chars=self.use_char_embeddings)
        self.processing_word_infer = get_processing_word(self.vocab_words,
                                                         self.vocab_chars,
                                                         lowercase=self.lowercase,
                                                         use_words=self.use_word_embeddings,
                                                         use_chars=self.use_char_embeddings)
        self.processing_tag = get_processing_tag(self.vocab_tags)

        # 3. get pre-trained embeddings
        self.embeddings = get_trimmed_glove_vectors(self.filename_embeddings_trimmed) if self.use_pretrained else None

        self.logger = get_logger(self.path_log)


class CoNLLDataset(BaseCoNLLDataset):
    def parse_line(self, line):
        ls = line.rsplit("\t", maxsplit=1)
        word, word_tags = ls[0], ls[-1].split('|')
        if self.processing_word is not None:
            word = self.processing_word(word)
        if self.processing_tag is not None:
            word_tags = self.processing_tag(word_tags)
        return word, word_tags


class DataBuilder(BaseDataBuilder):
    def __init__(self, config):
        super().__init__(config, CoNLLDataset)

    def handle_vocab_tags(self, vocab_tags_train, vocab_tags_dev, vocab_tags_test):
        vocab_tags = vocab_tags_train | vocab_tags_dev | vocab_tags_test
        for sym in [EOS, SOS, PAD, UNK]:
            if sym in vocab_tags:
                raise ValueError('Special symbol "%s" is already present in tag vocabulary' % sym)
        vocab_tags = [PAD, SOS, EOS, UNK] + list(vocab_tags)
        # create data output directory
        if not os.path.exists(self.config.out_data_dir):
            os.makedirs(self.config.out_data_dir)

        tag2idx_dict = create_tag_dict(vocab_tags)
        print("vocab_tags ({}): {}".format(len(vocab_tags), vocab_tags))
        print("tag2idx_dict: {}".format(tag2idx_dict))

        # Save vocab
        write_tag_dict(tag2idx_dict, self.config.filename_tags)


def labels2matrix(sequences, category_idx, max_sentence_length):
    """
    :param labels: batch of labels. For each word it contains a tuple (`category id`, `attribute id`)
    """
    assert isinstance(category_idx, (int, np.int32, np.int64))
    m = np.zeros([len(sequences), max_sentence_length],
                 dtype=np.int32)
    for i in range(len(sequences)):
        for j in range(len(sequences[i])):
            word_labels = sequences[i][j]
            m[i, j] = word_labels[category_idx]
    return m


def get_processing_tag(vocab_tags):
    """
    :param tag2idx:
    :param morph_categories:
    :return: list of ids for each category
    """

    def f(tags):
        """
        tags is a list of a form ["POS=Noun", "CASE=Nom", ....]
        """
        word_cat2tag_dict = {t.split("=")[0]: t for t in tags}
        tad_ids = []
        for cat in vocab_tags:
            if cat in word_cat2tag_dict:
                tid = vocab_tags[cat][word_cat2tag_dict[cat]]
            else:
                tid = vocab_tags[cat]["{}=NULL".format(cat)]
            tad_ids.append(tid)
        return tad_ids

    return f


def write_tag_dict(tag_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tag_dict, f)


def load_tag_dict(filename):
    with open(filename, 'rb') as f:
        tag_dict = pickle.load(f)
        return tag_dict


def create_tag_dict(tags):
    """
    :return: dict
        {category -> {tag -> tag-id}}
    """
    cat2tag_dict = defaultdict(set)
    for tag in tags:
        cat = tag.split("=")[0]
        cat2tag_dict[cat].add(tag)

    for cat, tag_set in cat2tag_dict.items():
        cat2tag_dict[cat] = ["{}=NULL".format(cat)] + list(tag_set)

    tag2idx = defaultdict(lambda: dict())
    for cat, tag_list in cat2tag_dict.items():
        for i, tag in enumerate(tag_list):
            tag2idx[cat][tag] = i
    tag2idx = OrderedDict(tag2idx)
    return tag2idx


def labels2one_hot(sequences, ntags):
    batch_size = len(sequences)
    max_sentence_length = max(len(s) for s in sequences)

    m = np.zeros((batch_size, max_sentence_length, ntags), dtype=np.int32)
    for i in range(len(sequences)):
        for j in range(len(sequences[i])):
            for k in sequences[i][j]:
                m[i, j, k] = 1.
    return m
