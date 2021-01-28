#!/usr/bin/env python

"""This script downloads the German-French Europarl parallel corpus
(http://opus.nlpl.eu/Europarl-v3.php), preprocesses it, and builds
from it a training, a validation and a test set, together with a
character vocabulary.

Reference:
J. Tiedemann, 2012, 'Parallel Data, Tools and Interfaces in OPUS',
in Proceedings of the 8th International Conference on Language Resources
and Evaluation (LREC 2012)
"""


import argparse
import itertools
import logging
import os
import pathlib
import random
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from typing import List, Dict, Tuple

from config import Config

logger = logging.getLogger(__name__)


# These constants are used in the function 'clean_sentence'. See the
# comments there for further explanation.
REPLACE_WITH_WHITESPACE = ["\xa0", "\N{SOFT HYPHEN}", "\N{NO-BREAK SPACE}"]
REMOVE_PREFIX = ["– ", "– ", "- ", ") ", "–"]
REMOVE_STRING = [
    "\xad",
    "(EN) ",
    "(FI) ",
    "(SV) ",
    "(NL) ",
    "(PT) ",
    "(IT) ",
    "(DE) ",
    "(EL) ",
    "(FR) ",
    "(ES) ",
    "(IL) ",
    "(DA) ",
]
FORBIDDEN_SUFFIXES = (")", "/", "]", ",", ", ?", " (")
FORBIDDEN_PREFIXES = (
    "Die Europäische Union",
    "Die Kommission",
    "Die Aussprache",
    "Le débat",
    "Le vote",
    "Die Abstimmung",
    "Rapport",
    "Bericht",
    "Anfrage",
    "question",
    "A5",
    "B5",
    "[",
    "(",
    ")",
    ".",
    ",",
    "?",
)


def replace_rare_chars(
    lines_src: List[str],
    lines_tgt: List[str],
    chars_special: List[str],
    char_unk: str,
    min_char_freq: float,
) -> List[str]:
    """Replaces rare characters with the 'unk'-character.

    All characters in the corpus which do not occur with the specified
    minimum frequency are replaced by the 'unk'-character.

    Args:
        lines_src: List of strings containing the source sentences.
        lines_tgt: List of strings containing the target sentences.
        chars_special: List containing the special characters.
        char_unk: The 'unk'-character.
        min_char_freq: The minimum frequency a character needs to have
            to not be replaced.

    Returns:
        A list of characters which occured frequently enough and
        have not been replaced (the special characters are always
        included).
    """
    # Count characters
    iter_all_sentences = itertools.chain(lines_src_cleaned, lines_tgt_cleaned)
    alphabet = Counter(itertools.chain.from_iterable(iter_all_sentences))
    for c in chars_special:
        alphabet[c] = -1

    # Determine which characters to keep and which to replace
    chars_total = sum(alphabet.values())
    replace_char = []
    for c in list(alphabet):
        if alphabet[c] / chars_total < min_char_freq and c not in chars_special:
            replace_char.append(c)
            del alphabet[c]
    replace_char += chars_special

    # Replace rare characters with 'char_unk'
    for line in range(len(lines_src_cleaned)):
        for c in replace_char:
            lines_src_cleaned[line] = lines_src_cleaned[line].replace(c, char_unk)
            lines_tgt_cleaned[line] = lines_tgt_cleaned[line].replace(c, char_unk)

    # Return a list of all characters which are not replaced
    return list(alphabet)


def create_dir(new_dir: str) -> None:
    """Creates a new directory with the specified name.

    Args:
        new_dir: Name of the directory to be created.
    """
    try:
        pathlib.Path(new_dir).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    else:
        logger.info("Create directory '{}'".format(new_dir))


def download_raw_corpus(
    files: Dict[str, str], urls: Dict[str, str], dir_dest: str
) -> Tuple[List[str], List[str]]:
    """Downloads the raw parallel corpus.

    All source and target files are assumed to be inside zip-files,
    which will be downloaded if necessary.

    Args:
        files: A dictionary containing the names of the source files as
            keys and the names of the target files as values.
        urls: A dictionary specifying for each file the url to the
            zip-file from which it can be extracted.
        dir_dest: Name of the directory where the files will be
            extracted.

    Returns:
        A list of source sentences and a list of target sentences.
    """
    remove_file: List[str] = []

    for f_raw, u_raw in urls.items():
        if not os.path.isfile(dir_dest + f_raw):
            f_raw_zip = u_raw[u_raw.rfind("/") + 1 :]

            # Download the zip-file if necessary
            if not os.path.isfile(dir_dest + f_raw_zip):
                logger.info("Download '{}' from '{}'".format(f_raw_zip, u_raw))
                remove_file.append(dir_dest + f_raw_zip)
                urllib.request.urlretrieve(u_raw, dir_dest + f_raw_zip)

            # Extract the desired file from the zip-file
            logger.info("Extract '{}' from '{}'".format(f_raw, f_raw_zip))
            with zipfile.ZipFile(dir_dest + f_raw_zip, "r") as zip_ref:
                zip_ref.extract(f_raw, path=dir_dest)

    # All downloaded zip-files will be deleted afterwards
    for f_raw_zip in remove_file:
        logger.info("Remove '{}'".format(f_raw_zip))
        os.remove(f_raw_zip)

    # Read the content of the files into lists
    lines_src: List[str] = []
    lines_tgt: List[str] = []

    for file_src, file_tgt in files.items():
        with open(dir_dest + file_src) as f_hdl_src:
            lines_src += f_hdl_src.readlines()

        with open(dir_dest + file_tgt) as f_hdl_tgt:
            lines_tgt += f_hdl_tgt.readlines()

    return lines_src, lines_tgt


def clean_corpus(
    lines_src: List[str], lines_tgt: List[str], max_len: int
) -> Tuple[List[str], List[str]]:
    """Cleans and filters the given parallel corpus.

    Args:
        lines_src: A list of source sentences.
        lines_tgt: A list of target sentences.
        max_len: The maximum sentence length.

    Returns:
        The cleaned list of source sentences and the cleaned list of
        target sentences.
    """
    lines_src_cleaned: List[str] = []
    lines_tgt_cleaned: List[str] = []

    # There are a lot of identical sentence pairs in the corpus. So we will
    # work through the raw sentence pairs in sorted order, and skip a pair
    # if its target sentence is identical to the target sentence of the
    # preceding pair.
    sorted_indices = list(range(len(lines_src)))
    sorted_indices.sort(key=lambda l: (len(lines_tgt[l]), lines_tgt[l]))

    last_idx = None
    for idx in sorted_indices:
        # Skip if the target sentence is equal to the target sentence of
        # the previous pair.
        if (
            last_idx is not None
            and lines_tgt[idx].strip() == lines_tgt[last_idx].strip()
        ):
            continue
        else:
            last_idx = idx

        # Clean the source and target sentences. Note that the function
        # 'clean_sentence' returns 'None' if the passed sentence is
        # deemed unsuitable.
        cleaned_line_src = clean_sentence(lines_src[idx], max_len)
        cleaned_line_tgt = clean_sentence(lines_tgt[idx], max_len)

        if cleaned_line_src is not None and cleaned_line_tgt is not None:
            lines_src_cleaned.append(cleaned_line_src)
            lines_tgt_cleaned.append(cleaned_line_tgt)

    # Shuffle everthing before returning the lists
    shuffled_idx = list(range(len(lines_src_cleaned)))
    random.shuffle(shuffled_idx)
    lines_src_cleaned = [lines_src_cleaned[i] for i in shuffled_idx]
    lines_tgt_cleaned = [lines_tgt_cleaned[i] for i in shuffled_idx]

    return lines_src_cleaned, lines_tgt_cleaned


def clean_sentence(sentence: str, max_len: int) -> str:
    """Cleans the given sentence.

    Certain sentences are unsuitable for training. In these cases the
    function returns 'None'.

    Note that most of the operations done here are specifically tailored
    towards the German-French Europarl parallel corpus.

    Args:
        sentence: The sentence to clean.
        max_len: The desired maximum sentence length.

    Returns:
        Either the cleaned sentence or 'None' if the sentence is
        deemed unsuitable.
    """
    sentence = unicodedata.normalize("NFKC", sentence)

    # Replace non-breaking space and other similar characters by usual
    # whitespace.
    for s in REPLACE_WITH_WHITESPACE:
        sentence = sentence.replace(s, " ")

    # Remove superfluous text fragments.
    for s in REMOVE_STRING:
        sentence = sentence.replace(s, "")

    for s in REMOVE_PREFIX:
        if sentence.startswith(s):
            sentence = sentence[len(s) :]

    sentence = sentence.strip()

    # A lot of small sentences in the corpus start almost identically.
    # This can influence the training negatively, so we simply ignore
    # sentence pairs which start with certain words. Furthermore,
    # sentences which start or end with certain characters often turn
    # out to be unsuitable for training, so we ignore them as well.
    if (
        sentence.startswith(FORBIDDEN_PREFIXES)
        or sentence.endswith(FORBIDDEN_SUFFIXES)
        or sentence[-1].isupper()
        or len(sentence) <= 2
        or len(sentence) > max_len
    ):
        return None

    return sentence


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # All default parameters are saved in the class 'Config'.
    conf = Config()

    # Names and URLs of the raw source and target files
    files_raw = conf.files_raw
    urls_raw = conf.urls_raw

    # The destination folder where the resulting files will be saved
    dir_dest_data = conf.dir_data

    # Names of the resulting files
    file_dest_alph = conf.file_alph
    files_dest_src = {
        "train": conf.file_train_src,
        "val": conf.file_val_src,
        "test": conf.file_test_src,
    }
    files_dest_tgt = {
        "train": conf.file_train_tgt,
        "val": conf.file_val_tgt,
        "test": conf.file_test_tgt,
    }

    # Our character vocabulary contains four special characters, among
    # them an 'unk'-character used to replace unknown characters.
    chars_special = list(conf.chars_special.values())
    char_unk = conf.chars_special["unknown"]

    # All characters which do not occur with a specified minimum
    # frequency will be replaced by the 'unk'-character.
    min_char_freq = conf.min_char_freq

    # We only consider sentences up to a certain length (we subtract 2
    # to acount for the fact that during training a 'start'- and
    # 'end'-character is added to each sentence).
    max_len = conf.max_len - 2

    # The desired sizes for each data set. A non-positive value for
    # 'sz_data_train' will be interpreted as no limit.
    sz_data_train = conf.size_data_train
    sz_data_val = conf.size_data_val
    sz_data_test = conf.size_data_test

    if sz_data_train > 0:
        sz_data = sz_data_train + sz_data_val + sz_data_test
    else:
        sz_data = -1

    # Start the actual preprocessing
    create_dir(dir_dest_data)
    lines_src, lines_tgt = download_raw_corpus(files_raw, urls_raw, dir_dest_data)

    if len(lines_src) != len(lines_tgt):
        logger.error("Error: File sizes don't match!")
        quit()
    else:
        logger.info("Total sentence pairs: {}".format(len(lines_src)))

    # Clean the downloaded corpus
    lines_src_cleaned, lines_tgt_cleaned = clean_corpus(lines_src, lines_tgt, max_len)

    logger.info("Suitable sentence pairs: {}".format(len(lines_src_cleaned)))

    if (sz_data_train < 0 and len(lines_src_cleaned) <= sz_data_val + sz_data_test) or (
        sz_data_train >= 0 and len(lines_src_cleaned) < sz_data
    ):
        logger.error("Error: Not enough data available for the requested sizes!")
        quit()

    # Check if there are enough sentence pairs
    if sz_data > 0:
        lines_src_cleaned = lines_src_cleaned[:sz_data]
        lines_tgt_cleaned = lines_tgt_cleaned[:sz_data]
    else:
        sz_data = len(lines_src_cleaned)

    # Replace all rar characters by the 'unk'-character
    alphabet = replace_rare_chars(
        lines_src_cleaned, lines_tgt_cleaned, chars_special, char_unk, min_char_freq
    )

    logger.info("Alphabet size: {}".format(len(alphabet)))

    # Write the alphabet to file
    with open(dir_dest_data + file_dest_alph, "w") as f_hdl_alph:
        f_hdl_alph.write("".join(alphabet))
        logger.info("Write alphabet to '{}'".format(dir_dest_data + file_dest_alph))

    # Write the train, validation and test data to file
    indices = {
        "test": list(range(sz_data_test)),
        "val": list(range(sz_data_test, sz_data_test + sz_data_val)),
        "train": list(range(sz_data_test + sz_data_val, sz_data)),
    }

    for d in {"test", "val", "train"}:
        filename_src = dir_dest_data + files_dest_src[d]
        filename_tgt = dir_dest_data + files_dest_tgt[d]

        logger.info(
            "Write {} sentence pairs to '{}' and '{}'".format(
                len(indices[d]), filename_src, filename_tgt
            )
        )

        with open(filename_src, "w") as f_hdl_src:
            for i in indices[d]:
                f_hdl_src.write(lines_src_cleaned[i] + "\n")

        with open(filename_tgt, "w") as f_hdl_tgt:
            for i in indices[d]:
                f_hdl_tgt.write(lines_tgt_cleaned[i] + "\n")
