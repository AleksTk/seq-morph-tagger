# Modeling Composite Labels for Neural Morphological Tagging

This repository contains a source code accompanying the paper
"Modeling Composite Labels for Neural Morphological Tagging", 2018
by Alexander Takchenko and Kairit Sirts.
The models are implemented in *python3* using *tensorflow 1.4*.

To reproduce experimental results, take the following steps:

* Install required packages:

        pip3 install -r requirements.txt

* Download [Universal Dependencies 2.1. dataset](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2515) and 
  convert *.conllu* files to *.ttm* format (see file *data/UD_Estonian/test.ttm*).
* Download [fasttext embeddings](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md).
* Organise the downloaded data into the following catalog tree:

        /seq-morph-tagger
            /code
                /...
            /data
                /UD_Estonian
                    /train.ttm
                    /test.ttm
                    /dev.ttm
                /UD_English
                    /...
                ...
            /embeddings
                /UD_Estonian
                    /emb.vec
                /UD_English
                    /...
                /...

* To train and evaluate a model use scripts
    
        ./train.sh
        ./test.sh

    as a template. By default, the scripts will train the *sequence* model for the Estonian dataset.


## Citation

    @InProceedings{tkachenko2018,
      author    = {Tkachenko, Alexander and Sirts, Kairit},
      title     = {Modeling Composite Labels for Neural Morphological Tagging},
      booktitle = {Proceedings of the 22nd Conference on Computational Natural Language Learning},
      year      = {2018}
    }
