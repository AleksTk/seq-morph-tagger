"""
Usage: test.py [--config=CONFIG-MODULE-PATH] [--output-dir=OUTPUT-DIR]
               [--data-dir=DATA-DIR] (--dev | --test) LANGUAGE-KEY

Options:
  --dev                      evaluate on development set
  --test                     evaluate on test set
  --output-dir=OUTPUT-DIR    model directory


Creates output files:
    $OUT_DIR/$LANGUAGE-KEY/evaluation.acc:
        Containing on row with values: lang_key, acc_morph_all, acc_morph_oov, acc_morph_voc, acc_pos_all, acc_pos_oov, acc_pos_voc

    $OUT_DIR/$LANGUAGE-KEY/predictions.csv:
        Contains all prediction for evaluated dataset.

"""
import os

from docopt import docopt

from mcml import ConfigHolder, Model
from utils import load_config_from_file
import evaluation


def predict_sentence(model, sentence_words, sentence_tags):
    preds = model.predict(sentence_words)
    y_true, y_pred = [], []
    for w, tt, pt in zip(sentence_words, sentence_tags, preds):
        tt = '|'.join(sorted(tt.split('|')))
        y_true.append(tt)
        pt = '|'.join(sorted(pt))
        y_pred.append(pt)
    return y_true, y_pred


if __name__ == "__main__":
    args = docopt(__doc__)
    lang_key = args['LANGUAGE-KEY']

    if args['--data-dir'] is None:
        os.environ["DATA_DIR"] = os.path.join(os.environ["DATA_DIR"], lang_key)
    else:
        os.environ["DATA_DIR"] = args['--data-dir']

    if args['--output-dir'] is None:
        os.environ["OUT_DIR"] = os.path.join(os.environ["OUT_DIR"], lang_key)
    else:
        os.environ["OUT_DIR"] = args['--output-dir']

    if args['--config'] is not None:
        config = load_config_from_file(args['--config'])
    else:
        from mcml import config

    print("Using configuration", config.__file__)

    if args['--test'] is True:
        test_file = config.filename_test
        eval_type = 'test'
    elif args['--dev'] is True:
        test_file = config.filename_dev
        eval_type = 'dev'
    else:
        raise ValueError('Specify --dev or --test.')

    config_holder = ConfigHolder(config)
    model = Model(config_holder)

    df = evaluation.predict(model, config_holder, test_file, predict_sentence_callback=predict_sentence)
    acc_dict = evaluation.calculate_accuracy(df)
    acc_verbose = evaluation.accuracy_to_string_verbose(acc_dict)
    evaluation.save_results(df, acc_dict, acc_verbose, lang_key, eval_type, config.out_dir)
