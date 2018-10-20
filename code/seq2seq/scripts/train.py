"""

Trains the sequence model.

Usage:

    train.py <LANGUAGE-ID>

"""
import os
import sys
import seq2seq

lang_key = sys.argv[1]

os.environ['OUT_DIR'] = os.path.join(os.environ['OUT_DIR'], lang_key)
os.environ['DATA_DIR'] = os.path.join(os.environ['DATA_DIR'], lang_key)
os.environ['EMBEDDINGS_DIR'] = os.path.join(os.environ['EMBEDDINGS_DIR'], lang_key)

import experiment_helper
from seq2seq import config

if os.path.exists(os.environ['OUT_DIR']):
    print("Output directory '%s' already exists. Exiting!" % os.environ['OUT_DIR'])
    sys.exit()

experiment_helper.run_experiment(seq2seq, config, title="Seq2seq %s" % lang_key)
