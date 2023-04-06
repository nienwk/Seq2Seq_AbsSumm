from typing import Iterable, Union, Sized
from ..dataset.AFFDataset import AmazonFineFoodDataset
from ..configs.dataloader_configs import BUCKET_MULTIPLIER
from torch.utils.data import Sampler
from torch import Tensor, Generator, randperm

class BucketSampler(Sampler):
    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        seed_generator: Union[Generator, None] = None,
        bucket_multiplier: int = BUCKET_MULTIPLIER,
        drop_last: bool = False,
        shuffle: bool = False,
        ) -> None:
        """Custom BucketSampler class.\n
        Yields batches of indices of which are drawn from sorted sub-buckets of the original dataset. Shuffles every epoch.

        E.g. Suppose we have original dataset of size 30. The following tensor will have the index, input_seq_len, summ_seq_len of each entry.
        tensor([[  0., 376.,  98.],
                [  1., 254., 100.],
                [  2., 339., 152.],
                [  3.,  33.,  91.],
                [  4., 152., 151.],
                [  5., 284., 162.],
                [  6.,  98., 173.],
                [  7., 245., 135.],
                [  8.,  79.,  71.],
                [  9., 197.,   1.],
                [ 10., 244.,  58.],
                [ 11., 157., 123.],
                [ 12., 351.,  59.],
                [ 13., 226., 155.],
                [ 14., 284., 168.],
                [ 15.,  14.,  78.],
                [ 16., 301.,  64.],
                [ 17., 174.,  48.],
                [ 18., 284.,  72.],
                [ 19.,   5., 128.],
                [ 20., 394., 184.],
                [ 21., 134., 104.],
                [ 22.,  58.,  71.],
                [ 23., 287.,  48.],
                [ 24., 136.,   2.],
                [ 25., 237.,  94.],
                [ 26.,  92.,  83.],
                [ 27., 171., 148.],
                [ 28., 133.,  58.],
                [ 29., 143., 126.]])

        Then assuming we have bucket size 3, we shuffle, sort by summ_len followed by sort by input_seq_len. Then we will have:
        tensor([[ 12., 351.,  59.],
                [  8.,  79.,  71.],
                [ 15.,  14.,  78.],
                [ 20., 394., 184.],
                [  2., 339., 152.],
                [  3.,  33.,  91.],
                [ 23., 287.,  48.],
                [ 27., 171., 148.],
                [ 19.,   5., 128.],
                [ 14., 284., 168.],
                [  7., 245., 135.],
                [ 29., 143., 126.],
                [  4., 152., 151.],
                [ 28., 133.,  58.],
                [ 22.,  58.,  71.],
                [ 13., 226., 155.],
                [ 11., 157., 123.],
                [ 21., 134., 104.],
                [ 25., 237.,  94.],
                [  9., 197.,   1.],
                [  6.,  98., 173.],
                [  5., 284., 162.],
                [ 10., 244.,  58.],
                [ 26.,  92.,  83.],
                [ 18., 284.,  72.],
                [  1., 254., 100.],
                [ 17., 174.,  48.],
                [  0., 376.,  98.],
                [ 16., 301.,  64.],
                [ 24., 136.,   2.]])
        
        We then yield sample indices in this order, based on batch size.\n
        Bucket size will be batch size multiplied by a bucket multiplier. Default multiplier is 100.
        """

        assert type(data_source) == AmazonFineFoodDataset, f"BucketSampler needs the dataset to implement sequence lengths getter function! See dataset/AmazonFineFoodDataset.py for example. Got {type(data_source)}"

        self.data_source = data_source
        self.batch_size = batch_size
        self.seed_generator = seed_generator
        self.bucket_multiplier = bucket_multiplier
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __len__(self) -> int:
        if len(self.data_source) % self.batch_size == 0:
            return int(len(self.data_source) // self.batch_size)
        else:
            return int(len(self.data_source) // self.batch_size) + 1
        

    def __iter__(self) -> Iterable:
        check = getattr(self.data_source, "get_seq_len") # Throws exception if not found
        if not callable(check):
            raise RuntimeError(f"Data source of BucketSampler needs to implement sequence lengths getter function. See dataset/AmazonFineFoodDataset.py for example. Got {type(self.data_source)}")

        # Convert original order (index, input_seq_len, summ_seq_len) to PyTorch Tensor
        idx_len_tuple_tensor = Tensor([(i,j,k) for i,(j,k) in enumerate(self.data_source.get_seq_len().to_numpy())])

        if self.shuffle:
            # Shuffle the indexes
            scramble = idx_len_tuple_tensor[randperm(len(idx_len_tuple_tensor), generator=self.seed_generator)]

            # Sort within each bucket of size (batch size * bucket multiplier)
            for i in range(0, len(scramble), self.batch_size * self.bucket_multiplier):
                bucket = scramble[i:i+(self.batch_size * self.bucket_multiplier)]
                tmp = bucket[bucket[:,2].sort(descending=True)[1]] #sort by index, value summ_len
                sorted_bucket = tmp[tmp[:,1].sort(descending=True,stable=True)[1]] #sort by index, value input_seq_len
                scramble[i:i+(self.batch_size * self.bucket_multiplier)] = sorted_bucket

            indices = scramble[:,0]

        else:
            # Choosing not to shuffle results in sorting the entire dataset as one big bucket
            tmp = idx_len_tuple_tensor[idx_len_tuple_tensor[:,2].sort(descending=True)[1]] #sort by index, value summ_len
            tmp = tmp[tmp[:,1].sort(descending=True,stable=True)[1]] #sort by index, value input_seq_len

            indices = tmp[:,0]

        for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if (len(batch) < self.batch_size) and self.drop_last: # drop last batch if it is too small
                    continue # can use break here as well, since it should only occur on the last batch
                yield batch.int().tolist()


