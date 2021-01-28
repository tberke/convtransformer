#!/usr/bin/env python

"""This script evaluates a trained convtransformer model on a test set
and computes the mean BLEU-4 score using SacreBLEU.

It expects three arguments:
    - The path to the trained model.
    - The file containing the source sentences of the test set.
    - The file containing the target sentences of the test set.

The beam width to be used in the beam search can be specified with the
flag --beam-width. All translated sentences can be saved in a separate
file which can be specified via the flag --write-to-file.

Example usage:
python evaluate.py --beam-width 5 model.pt test.src test.tgt
"""

import argparse
import os
import unicodedata
from sacrebleu import corpus_bleu
from tqdm import trange
from typing import List, Optional

import torch

from alphabet import Alphabet
from convtransformer import ConvTransformerModel
from translate import translate


DEFAULT_BEAM_WIDTH = 3


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a trained convtransformer model on a test set."
    )
    parser.add_argument("model", help="path to the model")
    parser.add_argument("source", help="test file with source sentences")
    parser.add_argument("target", help="test file with target sentences")
    parser.add_argument(
        "--write-to-file", help="save translations in the specified file"
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        help="set beam width (default: {})".format(DEFAULT_BEAM_WIDTH),
    )
    args = parser.parse_args()

    path_model = args.model
    file_src = args.source
    file_tgt = args.target

    if args.write_to_file:
        file_hyp = args.write_to_file
    else:
        file_hyp = None

    if args.beam_width:
        beam_width = args.beam_width
    else:
        beam_width = DEFAULT_BEAM_WIDTH

    # Check if model and data files exist
    for file_d in {path_model, file_src, file_tgt}:
        if not os.path.isfile(file_d):
            print("'{}' not found!".format(file_d))
            quit()

    # Load the model configuration states from the specified path
    print("Load model ({})".format(path_model))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    states = torch.load(path_model, map_location=device)

    conf_alph = states["conf_alph"]
    conf_model = states["conf_model"]
    model_state_dict = states["model_state_dict"]

    # Initialize the alphabet and the ConvTransformer model
    alph = Alphabet(**conf_alph)
    model = ConvTransformerModel(**conf_model).to(device)
    model.load_state_dict(model_state_dict)

    # Read test source and target files
    print("Load source file ({})".format(file_src))
    with open(file_src) as f_hdl:
        sentences_src = f_hdl.read().splitlines()

    print("Load target file ({})".format(file_tgt))
    with open(file_tgt) as f_hdl:
        sentences_tgt = f_hdl.read().splitlines()

    if len(sentences_src) != len(sentences_tgt):
        print("Number of sentences in source and target file don't match!")
        quit()

    nr_sent = len(sentences_src)
    print("{} Sentence Pairs".format(nr_sent))

    # If specified, we will save the translated sentences in a separate
    # file. We create here an empty file as placeholder to which the
    # translations will then be appended.
    if file_hyp is not None:
        print("Translated sentences will be written to '{}'.".format(file_hyp))
        with open(file_hyp, "w"):
            pass

    # Translate all sentences in the source file
    sentences_hyp = []
    print("Translate source sentences:")

    for i in trange(nr_sent, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        s, hyp = translate(sentences_src[i], alph, model, device, beam_width)
        sentences_hyp.append(hyp)

        if file_hyp is not None:
            with open(file_hyp, "a") as f_hdl:
                if i > 0:
                    f_hdl.write("\n")
                f_hdl.write(hyp)

    # We use SacreBLEU for computing the BLEU-score.
    bleu = corpus_bleu(sentences_hyp, [sentences_tgt])

    print("BLEU Score: {:1.2f}".format(bleu.score))
