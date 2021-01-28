from bidict import bidict
from typing import Dict, List, Optional


class Alphabet(bidict):
    """Bidirectional dictionary converting characters to indices.

    This is a helper class which maintains a lookup-table between
    characters and their associated indices in both directions.

    Each alphabet contains at least four special characters with special
    meanings:
        'char_pad': Used as padding character.
        'char_unk': Replacement for unknown characters.
        'char_start': Indicates the start of a sentence.
        'char_end': Indicates the end of a sentence.
    """

    def __init__(
        self,
        chars_special: Dict[str, str],
        from_file: Optional[str] = None,
        from_string: Optional[str] = None,
    ) -> None:
        """Initializes a new alphabet.

        Args:
            chars_special: Dictionary specifying which characters should
                be used for the special characters.
            from_file: If passed will add all characters in the
                specified file to the alphabet
            from_string: If passed will add all characters in the string
                to the alphabet.
        """
        super(Alphabet, self).__init__()

        self.char_pad = chars_special["padding"]
        self.char_start = chars_special["start"]
        self.char_end = chars_special["end"]
        self.char_unk = chars_special["unknown"]

        self[self.char_unk] = 0
        self[self.char_pad] = 1
        self[self.char_end] = 2
        self[self.char_start] = 3
        self.size = 4

        self.idx_pad = self[self.char_pad]
        self.idx_start = self[self.char_start]
        self.idx_end = self[self.char_end]
        self.idx_unk = self[self.char_unk]

        if from_file:
            with open(from_file) as f:
                chars = f.read()
            self.add(chars)

        if from_string:
            self.add(from_string)

    def add(self, string: str) -> None:
        """Adds the given characters to the alphabet.

        Args:
            string: String containing the characters to be added.
        """
        for c in string:
            if c not in self:
                self[c] = self.size
                self.size += 1

    def get_alph_as_str(self):
        """Returns a string containing all characters in the alphabet."""
        return "".join(self.keys())

    def str_to_idx(self, string: str) -> List[int]:
        """Converts the passed string to a list of indices.

        Characters not present in the alphabet will be replaced by the
        'unk'-character.

        Args:
            string: The string to be converted.

        Returns:
            List of indices representing the passed string.
        """
        idx: List[int] = []
        for c in string:
            if c not in self:
                idx += [self.idx_unk]
            else:
                idx += [self[c]]
        return idx

    def idx_to_str(self, idx: List[int]) -> str:
        """Converts a list of indices back to a string.

        All indices out of scope will be turned to an 'unk'-character.

        Args:
            idx: The list of indices to be converted.

        Returns:
            The resulting string.
        """
        string = ""
        for i in idx:
            if i < 0 or i >= self.size:
                string += self.char_unk
            else:
                string += self.inverse[i]
        return string

    def prepare_sentence(self, string: str) -> List[int]:
        """Converts a string into the required format for our model.

        All unknown characters are replaced by the 'unk'-character,
        'start'- and 'end'-character are added, and the string is
        converted to a list of indices.

        Args:
            string: The string to be prepared.

        Returns:
            List of indices in the required format.
        """
        string.replace(self.char_pad, self.char_unk)
        string.replace(self.char_start, self.char_unk)
        string.replace(self.char_end, self.char_unk)

        for i in range(len(string)):
            if string[i] not in self:
                string = string[:i] + self.char_unk + string[i + 1 :]
        return [self.idx_start] + self.str_to_idx(string) + [self.idx_end]

    def resize_sentence(self, sentence_idx: List[int], length: int) -> List[int]:
        """Resizes a sequence of indices to the desired length.

        If the sequence is too large, it will be cut. Otherwise padding
        indices will be added.

        Args:
            sentence_idx: List of indices representing the sentence.
            lentgh: The desired length of the sentence.

        Returns:
            Resized list of indices.
        """
        if len(sentence_idx) >= length:
            return sentence_idx[:length]
        else:
            return sentence_idx + [self.idx_pad] * (length - len(sentence_idx))
