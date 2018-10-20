from common_data_utils import *


class ConfigHolder(BaseConfigHolder):
    pass


class CoNLLDataset(BaseCoNLLDataset):
    def parse_line(self, line):
        ls = line.rsplit(maxsplit=1)
        word, tag = ls[0], ls[-1].split('|')

        if self.processing_word is not None:
            word = self.processing_word(word)
        if self.processing_tag is not None:
            tag = [self.processing_tag(t) for t in tag]
        return word, tag


class DataBuilder(BaseDataBuilder):
    def __init__(self, config):
        super().__init__(config, CoNLLDataset)
