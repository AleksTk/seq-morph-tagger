#!/usr/bin/env bash

LANGUAGE=UD_Estonian

export PYTHONPATH=code/

export DATA_DIR=data/
export EMBEDDINGS_DIR=embeddings/
export OUT_DIR=output/

python code/seq2seq/scripts/test.py --dev $LANGUAGE
