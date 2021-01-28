#!/usr/bin/env python

"""This script uses a trained convtransformer model to translate a
sentence from the source to the target language.

It expects two arguments:
    - The path to the trained model
    - The sentence to be translated

The beam width used in the beam search can be specified by using the
flag --beam-width.

Example usage:
python translate.py --beam-width 5 model.pt "Je parle allemand."
"""

import argparse
import os
from typing import List, Optional, Tuple

import torch
import torch.optim as optim

from alphabet import Alphabet
from convtransformer import ConvTransformerModel


DEFAULT_BEAM_WIDTH = 3


def translate(
    sentence: str,
    alph: Alphabet,
    model: ConvTransformerModel,
    device: torch.device,
    beam_width: int = DEFAULT_BEAM_WIDTH,
) -> Tuple[str, str]:
    """Translates a sentence using the passed model.

    We use beam search with beam width specified by 'beam_width'.

    Args:
        sentence: The sentence to be translated.
        alph: An instance of the class 'Alphabet'.
        model: A trained convtransformer model.
        device: The torch device to use.
        beam_width: The beam width in the beam search.

    Returns:
        The (possibly altered) source sentence and its translation.
    """
    if beam_width <= 0:
        raise ValueError("The beam width must be positive!")
    if beam_width > alph.size:
        raise ValueError("The beam width cannot be larger than the alphabet size!")

    # Turn the given sentence into a tensor
    sentence_idx = alph.prepare_sentence(sentence[: model.max_len - 2])
    src = torch.tensor(sentence_idx, device=device).unsqueeze(0)

    # In the list 'intermediate_results' we save all unfinished
    # translations considered at a given moment. Its elements are pairs,
    # where the first coefficient holds the log-probability of the
    # translation, and the second coefficient the translation itself.
    intermediate_results = [
        (torch.zeros(1).to(device), torch.tensor([alph.idx_start]).to(device))
    ]

    # In the list 'final_results' we save all finished translations. We
    # stop the search once we have found 'beam_width'**2 translations.
    final_results: List[Tuple[torch.Tensor, torch.Tensor]] = []

    model.eval()

    for current_tgt_length in range(1, model.max_len):
        # In the list 'next_sentences' we save all candidate
        # translations gathered in this round.
        next_sentences: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for log_prob, tgt in intermediate_results:
            # Given an unfinished translation 'tgt', get the
            # log-probability distribution for the next character
            with torch.no_grad():
                distr = model(src, tgt.unsqueeze(0))[1][0][-1]
            log_prob_topk, indices_topk = distr.topk(beam_width)

            # Add the best candidates either to 'next_sentences' (if
            # the translation is unfinished), or to 'final_results'
            # (if we encounter an 'end-'character).
            for i in range(beam_width):
                next_log_prob = log_prob + log_prob_topk[i]
                next_tgt = torch.cat([tgt, indices_topk[i].unsqueeze(0)])

                if (
                    indices_topk[i].item() == alph.idx_end
                    or current_tgt_length == model.max_len - 1
                ):
                    next_log_prob /= current_tgt_length + 1
                    final_results.append((next_log_prob, next_tgt))
                else:
                    next_sentences.append((next_log_prob, next_tgt))

        # Move the best candidate translations to 'intermediate_results'
        next_sentences.sort(key=lambda x: -x[0].item())
        intermediate_results = next_sentences[:beam_width]

        # Stop once we have enough finished translations
        if len(final_results) >= beam_width ** 2:
            break

    # Choose the translation with the highest (normalized) probability
    final_results.sort(key=lambda x: -x[0].item())
    translation_idx = final_results[0][1].tolist()

    # Turn the sentences back to strings
    sentence = alph.idx_to_str(sentence_idx)[1:-1]
    translation = alph.idx_to_str(translation_idx)[1:-1]

    return sentence, translation


if __name__ == "__main__":
    # Configure the argument parser
    parser = argparse.ArgumentParser(
        description="Translate a sentence using a trained convtransformer model."
    )
    parser.add_argument("model", help="path to the model")
    parser.add_argument("sentence", help="sentence to be translated")
    parser.add_argument(
        "--beam-width",
        type=int,
        help="set beam width (default: {})".format(DEFAULT_BEAM_WIDTH),
    )
    args = parser.parse_args()

    path_model = args.model
    original_sentence = args.sentence
    if args.beam_width:
        beam_width = args.beam_width
    else:
        beam_width = DEFAULT_BEAM_WIDTH

    # Load the model configuration states from the specified path
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if not os.path.isfile(path_model):
        print("Model not found!")
        quit()

    states = torch.load(path_model, map_location=device)
    conf_alph = states["conf_alph"]
    conf_model = states["conf_model"]
    model_state_dict = states["model_state_dict"]

    # Initialize the alphabet and the convtransformer model
    alph = Alphabet(**conf_alph)
    model = ConvTransformerModel(**conf_model).to(device)
    model.load_state_dict(model_state_dict)

    # Translate the given sentence
    sentence, translation = translate(
        original_sentence, alph, model, device, beam_width
    )

    print("Original:    {}".format(sentence))
    print("Translation: {}".format(translation))
