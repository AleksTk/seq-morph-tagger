"""
Trains the hmcml model.

Usage:

    train.py LANGUAGE-KEY

"""
import os
import sys
import mcml

lang_key = sys.argv[1]

os.environ['OUT_DIR'] = os.path.join(os.environ['OUT_DIR'], lang_key)
os.environ['DATA_DIR'] = os.path.join(os.environ['DATA_DIR'], lang_key)
os.environ['EMBEDDINGS_DIR'] = os.path.join(os.environ['EMBEDDINGS_DIR'], lang_key)

from hmcml import config

if os.path.exists(os.environ['OUT_DIR']):
    print("Output directory '%s' already exists. Exiting!" % os.environ['OUT_DIR'])
    sys.exit()

import experiment_helper

experiment_helper.run_experiment(mcml, config, title="HMCML %s" % lang_key)
