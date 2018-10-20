import os
from collections import OrderedDict
from time import time

from docopt import docopt
from sklearn.metrics import accuracy_score

import pandas as pd

from common_data_utils import minibatches
from utils import load_config_from_file
from mc import CoNLLDataset


def get_uniq_words(train_file, test_file):
    get_words = lambda _file: set([ln.split('\t')[0] for ln in open(_file, encoding='utf-8') if len(ln.rstrip()) > 0])
    train_words = get_words(train_file)
    test_words = get_words(test_file)
    unseen_words = test_words - train_words
    return unseen_words


def predict_sentence(model, sentence_words, sentence_tags):
    # Valid for seq2seq and mc
    preds = ['|'.join(p) if isinstance(p, (list, tuple)) else p
             for p in model.predict(sentence_words)]
    return sentence_tags, preds


def predict(model, config, test_file,
            predict_sentence_callback=predict_sentence):
    model.build()
    model.restore_session(config.dir_model)

    uniq_words = get_uniq_words(config.filename_train, test_file)
    print("Unique unseen words:", len(uniq_words))

    sentence_words, sentence_tags = [], []

    data = []
    evaluation_time = time()
    nsnts = 0 
    for line in open(test_file, encoding="utf-8"):
        line = line.rstrip()
        if line == "":
            if len(sentence_words) > 0:
                # end of sentence
                snt_y_true, snt_y_pred = predict_sentence_callback(model, sentence_words, sentence_tags)
                for w, tt, pt in zip(sentence_words, snt_y_true, snt_y_pred):
                    if w in uniq_words:
                        data.append((w, tt, pt, True, tt != pt))
                    else:
                        data.append((w, tt, pt, False, tt != pt))
                nsnts += 1
            sentence_words, sentence_tags = [], []
            
        else:
            items = line.rsplit("\t", maxsplit=1)
            word = items[0]
            tag = items[1]

            sentence_words.append(word)
            sentence_tags.append(tag)
    columns = ['word', 'tt_full', 'pt_full', 'uniq', 'err']
    df = pd.DataFrame(data=data, columns=columns)
    
    
    evaluation_time = time() - evaluation_time
    snts_per_sec = nsnts / float(evaluation_time)
    print("Evaluation time: %d sec; %d snt/sec  " % (evaluation_time, snts_per_sec))

    # set pos predictions
    df['tt_pos'] = df.tt_full.str.extract(r'POS=([A-Z]+)', expand=False).fillna('')
    df['pt_pos'] = df.pt_full.str.extract(r'POS=([A-Z]+)', expand=False).fillna('')

    # set morph predictions
    df['tt_morph'] = df.tt_full.str.replace(r'(\|POS=[A-Z]+\|)', '|').str.replace(r'(\|?POS=[A-Z]+\|?)', '')
    df['pt_morph'] = df.pt_full.str.replace(r'(\|POS=[A-Z]+\|)', '|').str.replace(r'(\|?POS=[A-Z]+\|?)', '')

    return df


def calculate_accuracy(df):
    return OrderedDict(
        # full tag  accuracy
        acc_full_all=accuracy_score(df.tt_full, df.pt_full),
        acc_full_oov=accuracy_score(df[df.uniq == True].tt_full, df[df.uniq == True].pt_full),
        acc_full_voc=accuracy_score(df[df.uniq == False].tt_full, df[df.uniq == False].pt_full),

        # pos accuracy
        acc_pos_all=accuracy_score(df.tt_pos, df.pt_pos),
        acc_pos_oov=accuracy_score(df[df.uniq == True].tt_pos, df[df.uniq == True].pt_pos),
        acc_pos_voc=accuracy_score(df[df.uniq == False].tt_pos, df[df.uniq == False].pt_pos),

        # morphology tag accuracy
        acc_morph_all=accuracy_score(df.tt_morph, df.pt_morph),
        acc_morph_oov=accuracy_score(df[df.uniq == True].tt_morph, df[df.uniq == True].pt_morph),
        acc_morph_voc=accuracy_score(df[df.uniq == False].tt_morph, df[df.uniq == False].pt_morph)
    )


def accuracy_to_string_verbose(acc_dict):
    return """
    FULL TAG ACCURACY:
    All words       : {acc_full_all}
    OOV words       : {acc_full_oov}
    Vocabulary words: {acc_full_voc}

    POS ACCURACY:
    All words       : {acc_pos_all}
    OOV words       : {acc_pos_oov}
    Vocabulary words: {acc_pos_voc}

    MORPH ACCURACY:
    All words       : {acc_morph_all}
    OOV words       : {acc_morph_oov}
    Vocabulary words: {acc_morph_voc}

    """.format(**acc_dict)


def save_results(df, acc_dict, acc_verbose, lang_key, eval_type, output_dir):
    # save verbose results
    print(acc_verbose)
    with open(os.path.join(output_dir, "evaluation.%s.log" % eval_type), "w") as f:
        print(acc_verbose, file=f)
        print("Saved evaluation result to", f.name)

    # save predictions
    predictions_file = os.path.join(output_dir, "predictions.%s.csv" % eval_type)
    df.to_csv(predictions_file, index=None)
    print("Saved predictions to", predictions_file)

    # save evaluation summary
    with open(os.path.join(output_dir, "evaluation.%s.acc" % eval_type), "w") as f:
        print(lang_key, *acc_dict.values(), sep=',', file=f)
        print("Saved evaluation results to", f.name)
