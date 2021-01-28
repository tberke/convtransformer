import random
from typing import Iterator, List, Tuple, Optional

import torch
from torch.utils.data import Sampler, Dataset, DataLoader

from alphabet import Alphabet


class ParallelDataset(Dataset):
    """Custom Dataset class to organize the parallel corpus.

    The actual sentence pairs are stored in the lists 'samples_src' and
    'samples_tgt'. On top of that we have a list 'buckets', where each
    bucket contains a list of indices corresponding to sentence pairs of
    similar length. Finally, we have a third list 'batches', which
    consists of 3-tuples of the form (bucket, start_idx, end_idx), where
    'start_idx' and 'end_idx' correspond to the start- and end-index in
    the respective bucket.

    During training, the function 'collate' creates from each such
    3-tuple the actual batch tensors.

    The list of 3-tuples, as well as each bucket is shuffled after each
    epoch.
    """

    def __init__(
        self,
        file_src: str,
        file_tgt: str,
        alph: Alphabet,
        max_len: int,
        sz_batch: Optional[int] = None,
        sz_bucket: Optional[int] = None,
    ) -> None:
        """Initializes the data set from the source and target files.

        The source and target files are assumed to be simple text files
        with one sentence per line.

        If 'sz_batch' is specified the batches are formed in such a way
        that the total number of characters in each batch does not
        exceed 'sz_batch'. In particular, 'sz_batch' must be at least as
        large as the maximum sequence length 'max_len'

        If any of 'sz_batch' or 'sz_bucket' is 'None', there will only
        be a single bucket and all batches will contain exactly one
        sentence pair.

        Args:
            file_src: The file containing the source sentences.
            file_tgt: The file containing the target sentences.
            alph: An instance of the 'Alphabet' class.
            max_len: The maximum sequence length.
            sz_batch: The maximum batch size (measured in characters).
            sz_bucket: The minimum bucket size (measured in sentence
                pairs).
        """
        if sz_batch is not None and sz_batch < max_len:
            raise ValueError(
                "Batch size has to be at least as large as the "
                "maximum sentence length!"
            )

        self.alph = alph

        # Load the parallel sentences from the specified files
        self.samples_src: List[List[int]] = []
        self.samples_tgt: List[List[int]] = []

        with open(file_src) as f_s, open(file_tgt) as f_t:

            line_src = f_s.readline()
            line_tgt = f_t.readline()

            while line_src and line_tgt:
                line_src = alph.prepare_sentence(line_src.strip())
                line_tgt = alph.prepare_sentence(line_tgt.strip())
                if len(line_src) <= max_len and len(line_tgt) <= max_len:
                    self.samples_src.append(line_src)
                    self.samples_tgt.append(line_tgt)
                line_src = f_s.readline()
                line_tgt = f_t.readline()

        if len(self.samples_src) != len(self.samples_tgt):
            raise RuntimeError("Number of source and target sentences don't "
                               "match!")

        # Determine at which sentence lengths to start a new bucket
        # (we want each bucket to hold at least 'sz_bucket' many
        # sentence pairs)
        bucket_points = [0]
        if sz_batch is not None and sz_bucket is not None:
            distr_len = [0] * (max_len + 1)
            for i in range(len(self.samples_src)):
                len_src = len(self.samples_src[i])
                len_tgt = len(self.samples_tgt[i])
                distr_len[max(len_src, len_tgt)] += 1

            sz_current_bucket = 0
            for i in range(max_len):
                sz_current_bucket += distr_len[i]
                if sz_current_bucket >= sz_bucket:
                    bucket_points += [i]
                    sz_current_bucket = 0
        bucket_points += [max_len]

        # Fill each bucket with the appropriate indices
        self.buckets = [[] for i in range(1, len(bucket_points))]

        which_bucket = [0]
        for i in range(1, len(bucket_points)):
            which_bucket += [i - 1] * (bucket_points[i] - bucket_points[i - 1])

        for i in range(len(self.samples_src)):
            len_src = len(self.samples_src[i])
            len_tgt = len(self.samples_tgt[i])
            self.buckets[which_bucket[max(len_src, len_tgt)]].append(i)

        # Create the actual batch 3-tuples
        self.batches: List[Tuple[int, int, int]] = []

        for bucket in range(len(self.buckets)):
            sz_current_bucket = len(self.buckets[bucket])
            sent_pairs_per_batch = self.get_pairs_per_batch(
                bucket_points[bucket + 1], sz_batch
            )
            for start_idx in range(0, sz_current_bucket, sent_pairs_per_batch):
                end_idx = min(start_idx + sent_pairs_per_batch, sz_current_bucket)
                self.batches.append((bucket, start_idx, end_idx))

        random.seed()
        self.shuffle()

        # We use a usual DataLoader to yield the batches.
        self.generator = DataLoader(
            self,
            batch_size=1,
            num_workers=0,
            sampler=ParallelSampler(self),
            collate_fn=self.collate,
        )

    def collate(
        self, batch: List[Tuple[int, int, int]]
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Creates the actual batch tensors from the passed 3-tuple.

        Args:
            batch: List holding exactly one 3-tuple of the form
                (bucket, start_idx, end_idx).

        Returns:
            A pair of batch tensors. The first tensor of shape (N, S)
            contains the source sentences, the second tensor of shape
            (N, T) contains the target sentences.
        """
        # The data loader always passes a batch of samples. Since we
        # specified a batch size of 1, the batch will always contain
        # exactly one element.
        bucket, start_idx, end_idx = batch[0]

        # Create the batches first in form of Python lists.
        batch_src: List[List[int]] = []
        batch_tgt: List[List[int]] = []
        for i in self.buckets[bucket][start_idx:end_idx]:
            batch_src.append(self.samples_src[i])
            batch_tgt.append(self.samples_tgt[i])

        # Insert padding to make all sentence have equal lenth.
        max_len_src = max([len(sent_idx) for sent_idx in batch_src])
        max_len_tgt = max([len(sent_idx) for sent_idx in batch_tgt])

        for i in range(len(batch_src)):
            batch_src[i] = self.alph.resize_sentence(batch_src[i], max_len_src)
            batch_tgt[i] = self.alph.resize_sentence(batch_tgt[i], max_len_tgt)

        return torch.tensor(batch_src), torch.tensor(batch_tgt)

    def shuffle(self) -> None:
        """Shuffles the data set.

        The indices in each bucket are shuffled, as well as the list
        containing the 3-tuples.
        """
        random.shuffle(self.batches)
        for b in self.buckets:
            random.shuffle(b)

    def get_pairs_per_batch(self, length: int, sz_batch: Optional[int] = None) -> int:
        """Calculates the number of sentence pairs that fit in a batch.

        Returns 1 if 'sz_batch' is 'None'. Otherwise returns the
        smallest power of 2 not larger than 'sz_batch/length'.

        Args:
            length: The maximum sentence length.
            sz_batch: The batch size (measured in characters).

        Returns:
            The maximum possible number of sentence pairs which fit into
            a batch of the specified size.
        """
        if sz_batch is None:
            return 1
        else:
            return 1 << ((sz_batch // length).bit_length() - 1)

    def __len__(self) -> int:
        """Returns the number of batches in the data set."""
        return len(self.batches)

    def __getitem__(self, index: int) -> Tuple[int, int, int]:
        """Returns the 3-tuple in 'batches' with the specified index."""
        return self.batches[index]


class ParallelSampler(Sampler):
    """Custom Sampler class to yield batches during training.

    The only difference to the standard SequentialSampler is that it
    shuffles the data set before starting a new iteration.
    """

    def __init__(self, dataset: ParallelDataset) -> None:
        """Initializes the sampler.

        Args:
            dataset: The data set to sample from.
        """
        self.dataset = dataset

    def __iter__(self) -> Iterator[int]:
        """Returns an iterator over the batch indices in the data set.

        The data set is shuffled before the iterator is created.
        """
        self.dataset.shuffle()
        return iter(range(len(self.dataset)))

    def __len__(self) -> int:
        """Returns the number of batches in the data set."""
        return len(self.dataset)
