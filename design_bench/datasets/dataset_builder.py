from design_bench.disk_resource import DiskResource
import numpy as np
import abc


def default_uniform_distribution(ranks):
    """Accepts the rank of a set of designs as input and returns an
    un-normalized uniform probability distribution

    Arguments:

    ranks: np.ndarray
        a numpy array representing the rank order of a set of designs given
        by their y values in a model-based optimization dataset

    Returns:

    probabilities: np.ndarray
        an un-normalized probability distribution that is passed to
        np.random.choice to subsample a model-based optimization dataset

    """

    return np.ones(ranks.shape, dtype=np.float32)


def default_linear_distribution(ranks):
    """Accepts the rank of a set of designs as input and returns an
    un-normalized linear probability distribution

    Arguments:

    ranks: np.ndarray
        a numpy array representing the rank order of a set of designs given
        by their y values in a model-based optimization dataset

    Returns:

    probabilities: np.ndarray
        an un-normalized probability distribution that is passed to
        np.random.choice to subsample a model-based optimization dataset

    """

    ranks = ranks.astype(np.float32)
    ranks = ranks / ranks.max()
    return 1.0 - ranks


def default_quadratic_distribution(ranks):
    """Accepts the rank of a set of designs as input and returns an
    un-normalized quadratic probability distribution

    Arguments:

    ranks: np.ndarray
        a numpy array representing the rank order of a set of designs given
        by their y values in a model-based optimization dataset

    Returns:

    probabilities: np.ndarray
        an un-normalized probability distribution that is passed to
        np.random.choice to subsample a model-based optimization dataset

    """

    ranks = ranks.astype(np.float32)
    ranks = ranks / ranks.max()
    return (1.0 - ranks)**2


def default_circular_distribution(ranks):
    """Accepts the rank of a set of designs as input and returns an
    un-normalized circular probability distribution

    Arguments:

    ranks: np.ndarray
        a numpy array representing the rank order of a set of designs given
        by their y values in a model-based optimization dataset

    Returns:

    probabilities: np.ndarray
        an un-normalized probability distribution that is passed to
        np.random.choice to subsample a model-based optimization dataset

    """

    ranks = ranks.astype(np.float32)
    ranks = ranks / ranks.max()
    return 1.0 - np.sqrt(1.0 - (ranks - 1.0)**2)


def default_exponential_distribution(ranks, c=3.0):
    """Accepts the rank of a set of designs as input and returns an
    un-normalized exponential probability distribution

    Arguments:

    ranks: np.ndarray
        a numpy array representing the rank order of a set of designs given
        by their y values in a model-based optimization dataset

    Returns:

    probabilities: np.ndarray
        an un-normalized probability distribution that is passed to
        np.random.choice to subsample a model-based optimization dataset

    """

    ranks = ranks.astype(np.float32)
    ranks = ranks / ranks.max()
    return np.exp(-c * ranks)


class DatasetBuilder(abc.ABC):
    """An abstract base class that defines a common set of functions
    and attributes for a model-based optimization dataset, where the
    goal is to find a design 'x' that maximizes a prediction 'y':

    max_x { y = f(x) }

    Public Attributes:

    name: str
        An attribute that specifies the name of a model-based optimization
        dataset, which might be used when labelling plots in a diagram of
        performance in a research paper using design-bench
    x_name: str
        An attribute that specifies the name of designs in a model-based
        optimization dataset, which might be used when labelling plots
        in a visualization of performance in a research paper
    y_name: str
        An attribute that specifies the name of predictions in a model-based
        optimization dataset, which might be used when labelling plots
        in a visualization of performance in a research paper

    x: np.ndarray
        the design values 'x' for a model-based optimization problem
        represented as a numpy array of arbitrary type
    input_shape: Tuple[int]
        the shape of a single design values 'x', represented as a list of
        integers similar to calling np.ndarray.shape
    input_size: int
        the total number of components in the design values 'x', represented
        as a single integer, the product of its shape entries
    input_dtype: np.dtype
        the data type of the design values 'x', which is typically either
        floating point or integer (np.float32 or np.int32)

    y: np.ndarray
        the prediction values 'y' for a model-based optimization problem
        represented by a scalar floating point value per 'x'
    output_shape: Tuple[int]
        the shape of a single prediction value 'y', represented as a list of
        integers similar to calling np.ndarray.shape
    output_size: int
        the total number of components in the prediction values 'y',
        represented as a single integer, the product of its shape entries
    output_dtype: np.dtype
        the data type of the prediction values 'y', which is typically a
        type of floating point (np.float32 or np.float16)

    dataset_size: int
        the total number of paired design values 'x' and prediction values
        'y' in the dataset, represented as a single integer
    dataset_distribution: Callable[np.ndarray, np.ndarray]
        the target distribution of the model-based optimization dataset
        marginal p(y) used for controlling the sampling distribution
    dataset_max_percentile: float
        the percentile between 0 and 100 of prediction values 'y' above
        which are hidden from access by members outside the class
    dataset_min_percentile: float
        the percentile between 0 and 100 of prediction values 'y' below
        which are hidden from access by members outside the class
    dataset_max_output: float
        the specific cutoff threshold for prediction values 'y' above
        which are hidden from access by members outside the class
    dataset_min_output: float
        the specific cutoff threshold for prediction values 'y' below
        which are hidden from access by members outside the class

    internal_batch_size: int
        the integer number of samples per batch that is used internally
        when processing the dataset and generating samples
    freeze_statistics: bool
        a boolean indicator that when set to true prevents methods from
        changing the normalization and sub sampling statistics

    is_normalized_x: bool
        a boolean indicator that specifies whether the design values
        in the dataset are being normalized
    x_mean: np.ndarray
        a numpy array that is automatically calculated to be the mean
        of visible design values in the dataset
    x_standard_dev: np.ndarray
        a numpy array that is automatically calculated to be the standard
        deviation of visible design values in the dataset

    is_normalized_y: bool
        a boolean indicator that specifies whether the prediction values
        in the dataset are being normalized
    y_mean: np.ndarray
        a numpy array that is automatically calculated to be the mean
        of visible prediction values in the dataset
    y_standard_dev: np.ndarray
        a numpy array that is automatically calculated to be the standard
        deviation of visible prediction values in the dataset

    Public Methods:

    iterate_batches(batch_size: int, return_x: bool,
                    return_y: bool, drop_remainder: bool)
                    -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model
    iterate_samples(return_x: bool, return_y: bool):
                    -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

    subsample(max_samples: int,
              distribution: Callable[np.ndarray, np.ndarray],
              max_percentile: float,
              min_percentile: float):
        a function that exposes a subsampled version of a much larger
        model-based optimization dataset containing design values 'x'
        whose prediction values 'y' are skewed
    relabel(relabel_function:
            Callable[[np.ndarray, np.ndarray], np.ndarray]):
        a function that accepts a function that maps from a dataset of
        design values 'x' and prediction values y to a new set of
        prediction values 'y' and relabels the model-based optimization dataset

    clone(subset: set, shard_size: int,
          to_disk: bool, disk_target: str, is_absolute: bool):
        Generate a cloned copy of a model-based optimization dataset
        using the provided name and shard generation settings; useful
        when relabelling a dataset buffer from the dis
    split(fraction: float, subset: set, shard_size: int,
          to_disk: bool, disk_target: str, is_absolute: bool):
        split a model-based optimization data set into a training set and
        a validation set allocating 'fraction' of the data set to the
        validation set and the rest to the training set

    normalize_x(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point design values 'x'
        as input and standardizes them so that they have zero
        empirical mean and unit empirical variance
    denormalize_x(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point design values 'x'
        as input and undoes standardization so that they have their
        original empirical mean and variance
    normalize_y(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point prediction values 'y'
        as input and standardizes them so that they have zero
        empirical mean and unit empirical variance
    denormalize_y(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point prediction values 'y'
        as input and undoes standardization so that they have their
        original empirical mean and variance

    map_normalize_x():
        a destructive function that standardizes the design values 'x'
        in the class dataset in-place so that they have zero empirical
        mean and unit variance
    map_denormalize_x():
        a destructive function that undoes standardization of the
        design values 'x' in the class dataset in-place which are expected
        to have zero  empirical mean and unit variance
    map_normalize_y():
        a destructive function that standardizes the prediction values 'y'
        in the class dataset in-place so that they have zero empirical
        mean and unit variance
    map_denormalize_y():
        a destructive function that undoes standardization of the
        prediction values 'y' in the class dataset in-place which are
        expected to have zero empirical mean and unit variance

    """

    @property
    @abc.abstractmethod
    def name(self):
        """Attribute that specifies the name of a model-based optimization
        dataset, which might be used when labelling plots in a diagram of
        performance in a research paper using design-bench

        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def x_name(self):
        """Attribute that specifies the name of designs in a model-based
        optimization dataset, which might be used when labelling plots
        in a visualization of performance in a research paper

        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def y_name(self):
        """Attribute that specifies the name of predictions in a model-based
        optimization dataset, which might be used when labelling plots
        in a visualization of performance in a research paper

        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def subclass_kwargs(self):
        """Generate a dictionary containing class initialization keyword
        arguments that are specific to sub classes; for example, may contain
        the number of classes in a discrete dataset

        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def subclass(self):
        """Specifies the primary subclass of an instance of DatasetBuilder
        that can be instantiated on its own using self.rebuild_dataset
        and typically either DiscreteDataset or ContinuousDataset

        """

        raise NotImplementedError

    def __init__(self, x_shards, y_shards, internal_batch_size=32,
                 is_normalized_x=False, is_normalized_y=False,
                 max_samples=None, distribution=None,
                 max_percentile=100.0, min_percentile=0.0):
        """Initialize a model-based optimization dataset and prepare
        that dataset by loading that dataset from disk and modifying
        its distribution of designs and predictions

        Arguments:

        x_shards: Union[         np.ndarray,           RemoteResource,
                        Iterable[np.ndarray], Iterable[RemoteResource]]
            a single shard or a list of shards representing the design values
            in a model-based optimization dataset; shards are loaded lazily
            if RemoteResource otherwise loaded in memory immediately
        y_shards: Union[         np.ndarray,           RemoteResource,
                        Iterable[np.ndarray], Iterable[RemoteResource]]
            a single shard or a list of shards representing prediction values
            in a model-based optimization dataset; shards are loaded lazily
            if RemoteResource otherwise loaded in memory immediately
        internal_batch_size: int
            the number of samples per batch to use when computing
            normalization statistics of the data set and while relabeling
            the prediction values of the data set
        is_normalized_x: bool
            a boolean indicator that specifies whether the designs
            in the dataset are being normalized
        is_normalized_y: bool
            a boolean indicator that specifies whether the predictions
            in the dataset are being normalized
        max_samples: int
            the maximum number of samples to include in the visible dataset;
            if more than this number of samples would be present, samples
            are randomly removed from the visible dataset
        distribution: Callable[np.ndarray, np.ndarray]
            a function that accepts an array of the ranks of designs as
            input and returns the probability to sample each according to
            a distribution---for example, a geometric distribution
        max_percentile: float
            the percentile between 0 and 100 of prediction values 'y' above
            which are hidden from access by members outside the class
        min_percentile: float
            the percentile between 0 and 100 of prediction values 'y' below
            which are hidden from access by members outside the class

        """

        # save the provided dataset shards to be loaded into batches
        self.x_shards = (x_shards,) if \
            isinstance(x_shards, np.ndarray) or \
            isinstance(x_shards, DiskResource) else x_shards
        self.y_shards = (y_shards,) if \
            isinstance(y_shards, np.ndarray) or \
            isinstance(y_shards, DiskResource) else y_shards

        # download the remote resources if they are given
        self.num_shards = 0
        for x_shard, y_shard in zip(self.x_shards, self.y_shards):
            self.num_shards += 1
            if isinstance(x_shard, DiskResource) \
                    and not x_shard.is_downloaded:
                x_shard.download()
            if isinstance(y_shard, DiskResource) \
                    and not y_shard.is_downloaded:
                y_shard.download()

        # update variables that describe the data set
        self.dataset_min_percentile = 0.0
        self.dataset_max_percentile = 100.0
        self.dataset_min_output = np.NINF
        self.dataset_max_output = np.PINF
        self.dataset_distribution = None

        # initialize the normalization state to False
        self.internal_batch_size = internal_batch_size
        self.is_normalized_x = False
        self.is_normalized_y = False

        # special flag that control when the dataset is mutable
        self.freeze_statistics = False
        self._disable_transform = False
        self._disable_subsample = False

        # initialize statistics for data set normalization
        self.x_mean = None
        self.y_mean = None
        self.x_standard_dev = None
        self.y_standard_dev = None

        # assign variables that describe the design values 'x'
        self._disable_transform = True
        self._disable_subsample = True
        for x in self.iterate_samples(return_y=False):
            self.input_shape = x.shape
            self.input_size = int(np.prod(x.shape))
            self.input_dtype = x.dtype
            break  # only sample a single design from the data set

        # assign variables that describe the prediction values 'y'
        self.output_shape = [1]
        self.output_size = 1
        self.output_dtype = np.float32

        # check the output format and count the number of samples
        self.dataset_size = 0
        for i, y in enumerate(self.iterate_samples(return_x=False)):
            self.dataset_size += 1  # assume the data set is large
            if i == 0 and len(y.shape) != 1 or y.shape[0] != 1:
                raise ValueError(f"predictions must have shape [N, 1]")

        # initialize a default set of visible designs
        self._disable_transform = False
        self._disable_subsample = False
        self.dataset_visible_mask = np.full(
            [self.dataset_size], True, dtype=np.bool_)

        # handle requests to normalize and subsample the dataset
        if is_normalized_x:
            self.map_normalize_x()
        if is_normalized_y:
            self.map_normalize_y()
            
        self.subsample(max_samples=max_samples,
                       distribution=distribution,
                       min_percentile=min_percentile,
                       max_percentile=max_percentile)

    def get_num_shards(self):
        """A helper function that returns the number of shards in a
        model-based optimization data set, which is useful when the data set
        is too large to be loaded inot memory all at once

        Returns:

        num_shards: int
            an integer representing the number of shards in a model-based
            optimization data set that can be loaded

        """

        return self.num_shards

    def get_shard_x(self, shard_id):
        """A helper function used for retrieving the data associated with a
        particular shard specified by shard_id containing design values
        in a model-based optimization data set

        Arguments:

        shard_id: int
            an integer representing the particular identifier of the shard
            to be loaded from a model-based optimization data set

        Returns:

        shard_data: np.ndarray
            a numpy array that represents the data encoded in the shard
            specified by the integer identifier shard_id

        """

        # check the shard id is in bounds
        if 0 < shard_id >= self.get_num_shards():
            raise ValueError(f"shard id={shard_id} out of bounds")

        # if that shard entry is a numpy array
        if isinstance(self.x_shards[shard_id], np.ndarray):
            return self.x_shards[shard_id]

        # if that shard entry is stored on the disk
        elif isinstance(self.x_shards[shard_id], DiskResource):
            return np.load(self.x_shards[shard_id].disk_target)

    def get_shard_y(self, shard_id):
        """A helper function used for retrieving the data associated with a
        particular shard specified by shard_id containing prediction values
        in a model-based optimization data set

        Arguments:

        shard_id: int
            an integer representing the particular identifier of the shard
            to be loaded from a model-based optimization data set

        Returns:

        shard_data: np.ndarray
            a numpy array that represents the data encoded in the shard
            specified by the integer identifier shard_id

        """

        # check the shard id is in bounds
        if 0 < shard_id >= self.get_num_shards():
            raise ValueError(f"shard id={shard_id} out of bounds")

        # if that shard entry is a numpy array
        if isinstance(self.y_shards[shard_id], np.ndarray):
            return self.y_shards[shard_id]

        # if that shard entry is stored on the disk
        elif isinstance(self.y_shards[shard_id], DiskResource):
            return np.load(self.y_shards[shard_id].disk_target)

    def set_shard_x(self, shard_id, shard_data,
                    to_disk=None, disk_target=None, is_absolute=None):
        """A helper function used for assigning the data associated with a
        particular shard specified by shard_id containing design values
        in a model-based optimization data set

        Arguments:

        shard_id: int
            an integer representing the particular identifier of the shard
            to be loaded from a model-based optimization data set
        shard_data: np.ndarray
            a numpy array that represents the data to be encoded in the
            shard specified by the integer identifier shard_id
        to_disk: boolean
            a boolean that indicates whether to store the data set
            in memory as numpy arrays or to the disk
        disk_target: str
            a string that determines the name and sub folder of the saved
            data set if to_disk is set to be true
        is_absolute: boolean
            a boolean that indicates whether the disk_target path is taken
            relative to the benchmark data folder

        """

        # check that all arguments are set when saving to disk
        if to_disk is not None and to_disk and \
                (disk_target is None or is_absolute is None):
            raise ValueError("must specify location when saving to disk")

        # check the shard id is in bounds
        if 0 < shard_id >= self.get_num_shards():
            raise ValueError(f"shard id={shard_id} out of bounds")

        # store shard in memory as a numpy array
        if (to_disk is not None and not to_disk) or \
                (to_disk is None and isinstance(
                    self.x_shards[shard_id], np.ndarray)):
            self.x_shards[shard_id] = shard_data

        # write shard to a new resource file given by "disk_target"
        if to_disk is not None and to_disk:
            disk_target = f"{disk_target}-x-{shard_id}.npy"
            self.x_shards[shard_id] = DiskResource(disk_target,
                                                   is_absolute=is_absolute)

        # possibly write shard to an existing file on disk
        if isinstance(self.x_shards[shard_id], DiskResource):
            np.save(self.x_shards[shard_id].disk_target, shard_data)

    def set_shard_y(self, shard_id, shard_data,
                    to_disk=None, disk_target=None, is_absolute=None):
        """A helper function used for assigning the data associated with a
        particular shard specified by shard_id containing prediction values
        in a model-based optimization data set

        Arguments:

        shard_id: int
            an integer representing the particular identifier of the shard
            to be loaded from a model-based optimization data set
        shard_data: np.ndarray
            a numpy array that represents the data to be encoded in the
            shard specified by the integer identifier shard_id
        to_disk: boolean
            a boolean that indicates whether to store the data set
            in memory as numpy arrays or to the disk
        disk_target: str
            a string that determines the name and sub folder of the saved
            data set if to_disk is set to be true
        is_absolute: boolean
            a boolean that indicates whether the disk_target path is taken
            relative to the benchmark data folder

        """

        # check that all arguments are set when saving to disk
        if to_disk is not None and to_disk and \
                (disk_target is None or is_absolute is None):
            raise ValueError("must specify location when saving to disk")

        # check the shard id is in bounds
        if 0 < shard_id >= self.get_num_shards():
            raise ValueError(f"shard id={shard_id} out of bounds")

        # store shard in memory as a numpy array
        if (to_disk is not None and not to_disk) or \
                (to_disk is None and isinstance(
                    self.y_shards[shard_id], np.ndarray)):
            self.y_shards[shard_id] = shard_data

        # write shard to a new resource file given by "disk_target"
        if to_disk is not None and to_disk:
            disk_target = f"{disk_target}-y-{shard_id}.npy"
            self.y_shards[shard_id] = DiskResource(disk_target,
                                                   is_absolute=is_absolute)

        # possibly write shard to an existing file on disk
        if isinstance(self.y_shards[shard_id], DiskResource):
            np.save(self.y_shards[shard_id].disk_target, shard_data)

    def batch_transform(self, x_batch, y_batch,
                        return_x=True, return_y=True):
        """Apply a transformation to batches of samples from a model-based
        optimization data set, including sub sampling and normalization
        and potentially other used defined transformations

        Arguments:

        x_batch: np.ndarray
            a numpy array representing a batch of design values sampled
            from a model-based optimization data set
        y_batch: np.ndarray
            a numpy array representing a batch of prediction values sampled
            from a model-based optimization data set
        return_x: bool
            a boolean indicator that specifies whether the generator yields
            design values at every iteration; note that at least one of
            return_x and return_y must be set to True
        return_y: bool
            a boolean indicator that specifies whether the generator yields
            prediction values at every iteration; note that at least one
            of return_x and return_y must be set to True

        Returns:

        x_batch: np.ndarray
            a numpy array representing a batch of design values sampled
            from a model-based optimization data set
        y_batch: np.ndarray
            a numpy array representing a batch of prediction values sampled
            from a model-based optimization data set

        """

        # normalize the design values in the batch
        if self.is_normalized_x and return_x:
            x_batch = self.normalize_x(x_batch)

        # normalize the prediction values in the batch
        if self.is_normalized_y and return_y:
            y_batch = self.normalize_y(y_batch)

        # return processed batches of designs an predictions
        return (x_batch if return_x else None,
                y_batch if return_y else None)

    def iterate_batches(self, batch_size, return_x=True,
                        return_y=True, drop_remainder=False):
        """Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

        Arguments:

        batch_size: int
            a positive integer that specifies the batch size of samples
            taken from a model-based optimization data set; batches
            with batch_size elements are yielded
        return_x: bool
            a boolean indicator that specifies whether the generator yields
            design values at every iteration; note that at least one of
            return_x and return_y must be set to True
        return_y: bool
            a boolean indicator that specifies whether the generator yields
            prediction values at every iteration; note that at least one
            of return_x and return_y must be set to True
        drop_remainder: bool
            a boolean indicator representing whether the last batch
            should be dropped in the case it has fewer than batch_size
            elements; the default behavior is not to drop the smaller batch.

        Returns:

        generator: Iterator
            a python iterable that yields samples from a model-based
            optimization data set and returns once finished

        """

        # check whether the generator arguments are valid
        if batch_size < 1 or (not return_x and not return_y):
            raise ValueError("invalid arguments passed to batch generator")

        # track a list of incomplete batches between shards
        y_batch_size = 0
        x_batch = [] if return_x else None
        y_batch = [] if return_y else None

        # iterate through every registered shard
        sample_id = 0
        for shard_id in range(self.get_num_shards()):
            x_shard_data = self.get_shard_x(shard_id) if return_x else None
            y_shard_data = self.get_shard_y(shard_id)

            # loop once per batch contained in the shard
            shard_position = 0
            while shard_position < y_shard_data.shape[0]:

                # how many samples will be attempted to read
                target_size = batch_size - y_batch_size

                # slice out a component of the current shard
                x_sliced = x_shard_data[shard_position:(
                    shard_position + target_size)] if return_x else None
                y_sliced = y_shard_data[shard_position:(
                    shard_position + target_size)]

                # store the batch_size of samples read
                samples_read = y_sliced.shape[0]

                # take a subset of the sliced arrays using a pre-defined
                # transformation that sub-samples
                if not self._disable_subsample:

                    # compute which samples are exposed in the dataset
                    indices = np.where(self.dataset_visible_mask[
                        sample_id:sample_id + y_sliced.shape[0]])[0]

                    # sub sample the design and prediction values
                    x_sliced = x_sliced[indices] if return_x else None
                    y_sliced = y_sliced[indices] if return_y else None

                # take a subset of the sliced arrays using a pre-defined
                # transformation that normalizes
                if not self._disable_transform:

                    # apply a transformation to the dataset
                    x_sliced, y_sliced = self.batch_transform(
                        x_sliced, y_sliced,
                        return_x=return_x, return_y=return_y)

                # update the read position in the shard tensor
                shard_position += target_size
                sample_id += samples_read

                # update the current batch to be yielded
                y_batch_size += (y_sliced if
                                 return_y else x_sliced).shape[0]
                x_batch.append(x_sliced) if return_x else None
                y_batch.append(y_sliced) if return_y else None

                # yield the current batch when enough samples are loaded
                if y_batch_size >= batch_size \
                        or (shard_position >= y_shard_data.shape[0]
                            and shard_id + 1 == self.get_num_shards()
                            and not drop_remainder):

                    try:

                        # determine which tensors to yield
                        if return_x and return_y:
                            yield np.concatenate(x_batch, axis=0), \
                                  np.concatenate(y_batch, axis=0)
                        elif return_x:
                            yield np.concatenate(x_batch, axis=0)
                        elif return_y:
                            yield np.concatenate(y_batch, axis=0)

                        # reset the buffer for incomplete batches
                        y_batch_size = 0
                        x_batch = [] if return_x else None
                        y_batch = [] if return_y else None

                    except GeneratorExit:

                        # handle cleanup when break is called
                        return

    def iterate_samples(self, return_x=True, return_y=True):
        """Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

        Arguments:

        return_x: bool
            a boolean indicator that specifies whether the generator yields
            design values at every iteration; note that at least one of
            return_x and return_y must be set to True
        return_y: bool
            a boolean indicator that specifies whether the generator yields
            prediction values at every iteration; note that at least one
            of return_x and return_y must be set to True

        Returns:

        generator: Iterator
            a python iterable that yields samples from a model-based
            optimization data set and returns once finished

        """

        # generator that only returns single samples
        for batch in self.iterate_batches(
                self.internal_batch_size,
                return_x=return_x, return_y=return_y):

            # yield a tuple if both x and y are returned
            if return_x and return_y:
                for i in range(batch[0].shape[0]):
                    yield batch[0][i], batch[1][i]

            # yield a tuple if only x and y or returned
            elif return_x or return_y:
                for i in range(batch.shape[0]):
                    yield batch[i]

    def __iter__(self):
        """Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

        Returns:

        generator: Iterator
            a python iterable that yields samples from a model-based
            optimization data set and returns once finished

        """

        # generator that returns batches of designs and predictions
        for x_batch, y_batch in \
                self.iterate_batches(self.internal_batch_size):
            yield x_batch, y_batch

    def update_x_statistics(self):
        """A helpful function that calculates the mean and standard deviation
        of the designs and predictions in a model-based optimization dataset
        either iteratively or all at once using numpy

        """

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # make sure the statistics are calculated from original samples
        original_is_normalized_x = self.is_normalized_x
        self.is_normalized_x = False

        # iterate through the entire dataset a first time
        samples = x_mean = 0
        for x_batch in self.iterate_batches(
                self.internal_batch_size, return_y=False):

            # calculate how many samples are actually in the current batch
            batch_size = np.array(x_batch.shape[0], dtype=np.float32)

            # update the running mean using dynamic programming
            x_mean = x_mean * (samples / (samples + batch_size)) + \
                np.sum(x_batch,
                       axis=0, keepdims=True) / (samples + batch_size)

            # update the number of samples used in the calculation
            samples += batch_size

        # iterate through the entire dataset a second time
        samples = x_variance = 0
        for x_batch in self.iterate_batches(
                self.internal_batch_size, return_y=False):

            # calculate how many samples are actually in the current batch
            batch_size = np.array(x_batch.shape[0], dtype=np.float32)

            # update the running variance using dynamic programming
            x_variance = x_variance * (samples / (samples + batch_size)) + \
                np.sum(np.square(x_batch - x_mean),
                       axis=0, keepdims=True) / (samples + batch_size)

            # update the number of samples used in the calculation
            samples += batch_size

        # expose the calculated mean and standard deviation
        self.x_mean = x_mean
        self.x_standard_dev = np.sqrt(x_variance)

        # remove zero standard deviations to prevent singularities
        self.x_standard_dev = np.where(
            self.x_standard_dev == 0.0, 1.0, self.x_standard_dev)

        # reset the normalized state to what it originally was
        self.is_normalized_x = original_is_normalized_x

    def update_y_statistics(self):
        """A helpful function that calculates the mean and standard deviation
        of the designs and predictions in a model-based optimization dataset
        either iteratively or all at once using numpy

        """

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # make sure the statistics are calculated from original samples
        original_is_normalized_y = self.is_normalized_y
        self.is_normalized_y = False

        # iterate through the entire dataset a first time
        samples = y_mean = 0
        for y_batch in self.iterate_batches(
                self.internal_batch_size, return_x=False):

            # calculate how many samples are actually in the current batch
            batch_size = np.array(y_batch.shape[0], dtype=np.float32)

            # update the running mean using dynamic programming
            y_mean = y_mean * (samples / (samples + batch_size)) + \
                np.sum(y_batch,
                       axis=0, keepdims=True) / (samples + batch_size)

            # update the number of samples used in the calculation
            samples += batch_size

        # iterate through the entire dataset a second time
        samples = y_variance = 0
        for y_batch in self.iterate_batches(
                self.internal_batch_size, return_x=False):

            # calculate how many samples are actually in the current batch
            batch_size = np.array(y_batch.shape[0], dtype=np.float32)

            # update the running variance using dynamic programming
            y_variance = y_variance * (samples / (samples + batch_size)) + \
                np.sum(np.square(y_batch - y_mean),
                       axis=0, keepdims=True) / (samples + batch_size)

            # update the number of samples used in the calculation
            samples += batch_size

        # expose the calculated mean and standard deviation
        self.y_mean = y_mean
        self.y_standard_dev = np.sqrt(y_variance)

        # remove zero standard deviations to prevent singularities
        self.y_standard_dev = np.where(
            self.y_standard_dev == 0.0, 1.0, self.y_standard_dev)

        # reset the normalized state to what it originally was
        self.is_normalized_y = original_is_normalized_y

    def subsample(self, max_samples=None, distribution=None,
                  max_percentile=100.0, min_percentile=0.0):
        """a function that exposes a subsampled version of a much larger
        model-based optimization dataset containing design values 'x'
        whose prediction values 'y' are skewed

        Arguments:

        max_samples: int
            the maximum number of samples to include in the visible dataset;
            if more than this number of samples would be present, samples
            are randomly removed from the visible dataset
        distribution: Callable[np.ndarray, np.ndarray]
            a function that accepts an array of the ranks of designs as
            input and returns the probability to sample each according to
            a distribution---for example, a geometric distribution
        max_percentile: float
            the percentile between 0 and 100 of prediction values 'y' above
            which are hidden from access by members outside the class
        min_percentile: float
            the percentile between 0 and 100 of prediction values 'y' below
            which are hidden from access by members outside the class

        """

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # return an error is the arguments are invalid
        if max_samples is not None and max_samples <= 0:
            raise ValueError("dataset cannot be made empty")

        # return an error is the arguments are invalid
        if min_percentile > max_percentile:
            raise ValueError("invalid arguments provided")

        # convert the original prediction generator to a numpy tensor
        self._disable_subsample = True
        self._disable_transform = True
        y = np.concatenate(list(self.iterate_batches(
            self.internal_batch_size, return_x=False)), axis=0)
        self._disable_subsample = False
        self._disable_transform = False

        # calculate the min threshold for predictions in the dataset
        min_output = np.percentile(y[:, 0], min_percentile) \
            if min_percentile > 0.0 else np.NINF
        self.dataset_min_percentile = min_percentile
        self.dataset_min_output = min_output

        # calculate the max threshold for predictions in the dataset
        max_output = np.percentile(y[:, 0], max_percentile) \
            if max_percentile < 100.0 else np.PINF
        self.dataset_max_percentile = max_percentile
        self.dataset_max_output = max_output

        # calculate indices of samples that are within range
        indices = np.arange(y.shape[0])[np.where(
            np.logical_and(y <= max_output, y >= min_output))[0]]
        max_samples = indices.size \
            if max_samples is None else min(indices.size, max_samples)

        # replace default distributions with their implementations
        if distribution in {None, "uniform"}:
            distribution = default_uniform_distribution
        elif distribution == "linear":
            distribution = default_linear_distribution
        elif distribution == "quadratic":
            distribution = default_quadratic_distribution
        elif distribution == "exponential":
            distribution = default_exponential_distribution
        elif distribution == "circular":
            distribution = default_circular_distribution

        # calculate the probability to subsample individual designs
        probs = distribution(y[indices, 0].argsort().argsort())
        probs = np.asarray(probs, dtype=np.float32)
        probs = np.broadcast_to(probs, (indices.size,))
        indices = indices[np.random.choice(
            indices.size, max_samples, replace=False, p=probs / probs.sum())]

        # binary mask that determines which samples are visible
        visible_mask = np.full([y.shape[0]], False, dtype=np.bool_)
        visible_mask[indices] = True
        self.dataset_visible_mask = visible_mask
        self.dataset_size = indices.size
        self.dataset_distribution = distribution

        # update normalization statistics for design values
        if self.is_normalized_x:
            self.update_x_statistics()

        # update normalization statistics for prediction values
        if self.is_normalized_y:
            self.update_y_statistics()

    @property
    def x(self) -> np.ndarray:
        """A helpful function for loading the design values from disk in case
        the dataset is set to load all at once rather than lazily and is
        overridden with a numpy array once loaded

        Returns:

        x: np.ndarray
            processed design values 'x' for a model-based optimization problem
            represented as a numpy array of arbitrary type

        """

        return np.concatenate([x for x in self.iterate_batches(
            self.internal_batch_size, return_y=False)], axis=0)

    @property
    def y(self) -> np.ndarray:
        """A helpful function for loading prediction values from disk in case
        the dataset is set to load all at once rather than lazily and is
        overridden with a numpy array once loaded

        Returns:

        y: np.ndarray
            processed prediction values 'y' for a model-based optimization
            problem represented as a numpy array of arbitrary type

        """

        return np.concatenate([y for y in self.iterate_batches(
            self.internal_batch_size, return_x=False)], axis=0)

    def relabel(self, relabel_function,
                to_disk=None, disk_target=None, is_absolute=None):
        """a function that accepts a function that maps from a dataset of
        design values 'x' and prediction values y to a new set of
        prediction values 'y' and relabels a model-based optimization dataset

        Arguments:

        relabel_function: Callable[[np.ndarray, np.ndarray], np.ndarray]
            a function capable of mapping from a numpy array of design
            values 'x' and prediction values 'y' to new predictions 'y' 
            using batching to prevent memory overflow
        to_disk: boolean
            a boolean that indicates whether to store the data set
            in memory as numpy arrays or to the disk
        disk_target: str
            a string that determines the name and sub folder of the saved
            data set if to_disk is set to be true
        is_absolute: boolean
            a boolean that indicates whether the disk_target path is taken
            relative to the benchmark data folder

        """

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check that all arguments are set when saving to disk
        if to_disk is not None and to_disk and \
                (disk_target is None or is_absolute is None):
            raise ValueError("must specify location when saving to disk")

        # prevent the data set for being sub-sampled or normalized
        self._disable_subsample = True
        examples = self.y.shape[0]
        examples_processed = 0

        # track a list of incomplete batches between shards
        y_shard = []
        y_shard_size = 0

        # calculate the appropriate size of the first shard
        shard_id = 0
        shard = self.get_shard_y(shard_id)
        shard_size = shard.shape[0]

        # relabel the prediction values of the internal data set
        for x_batch, y_batch in \
                self.iterate_batches(self.internal_batch_size):

            # calculate the new prediction values to be stored as shards
            y_batch = relabel_function(x_batch, y_batch)
            read_position = 0

            # remove potential normalization on the predictions
            if self.is_normalized_y:
                y_batch = self.denormalize_y(y_batch)

            # loop once per batch contained in the shard
            while read_position < y_batch.shape[0]:

                # calculate the intended number of samples to serialize
                target_size = shard_size - y_shard_size

                # slice out a component of the current shard
                y_slice = y_batch[read_position:read_position + target_size]
                samples_read = y_slice.shape[0]

                # increment the read position in the prediction tensor
                # and update the number of shards and examples processed
                read_position += target_size
                examples_processed += samples_read

                # update the current shard to be serialized
                y_shard.append(y_slice)
                y_shard_size += samples_read

                # yield the current batch when enough samples are loaded
                if y_shard_size >= shard_size \
                        or examples_processed >= examples:

                    # serialize the value of the new shard data
                    self.set_shard_y(shard_id, np.concatenate(y_shard, axis=0),
                                     to_disk=to_disk, disk_target=disk_target,
                                     is_absolute=is_absolute)

                    # reset the buffer for incomplete batches
                    y_shard = []
                    y_shard_size = 0

                    # calculate the appropriate size for the next shard
                    if not examples_processed >= examples:
                        shard_id += 1
                        shard = self.get_shard_y(shard_id)
                        shard_size = shard.shape[0]

        # re-sample the data set and recalculate statistics
        self._disable_subsample = False
        self.subsample(max_samples=self.dataset_size,
                       distribution=self.dataset_distribution,
                       max_percentile=self.dataset_max_percentile,
                       min_percentile=self.dataset_min_percentile)

    def rebuild_dataset(self, x_shards, y_shards, visible_mask):
        """Initialize a model-based optimization dataset and prepare
        that dataset by loading that dataset from disk and modifying
        its distribution of designs and predictions

        Arguments:

        x_shards: Union[         np.ndarray,           RemoteResource,
                        Iterable[np.ndarray], Iterable[RemoteResource]]
            a single shard or a list of shards representing the design values
            in a model-based optimization dataset; shards are loaded lazily
            if RemoteResource otherwise loaded in memory immediately
        y_shards: Union[         np.ndarray,           RemoteResource,
                        Iterable[np.ndarray], Iterable[RemoteResource]]
            a single shard or a list of shards representing prediction values
            in a model-based optimization dataset; shards are loaded lazily
            if RemoteResource otherwise loaded in memory immediately
        visible_mask: np.ndarray
            a numpy array of shape [dataset_size] containing boolean entries
            specifying which samples are visible in the provided Iterable

        Returns:

        dataset: DatasetBuilder
            an instance of a data set builder subclass containing a copy
            of all statistics associated with this dataset

        """

        # new dataset that shares statistics with this one
        kwargs = dict(internal_batch_size=self.internal_batch_size)
        kwargs.update(self.subclass_kwargs)
        dataset = self.subclass(x_shards, y_shards, **kwargs)

        # carry over the names of the parent
        dataset.name = self.name
        dataset.x_name = self.x_name
        dataset.y_name = self.y_name

        # carry over the normalize statistics of the parent
        dataset.is_normalized_x = self.is_normalized_x
        dataset.x_mean = self.x_mean
        dataset.x_standard_dev = self.x_standard_dev

        # carry over the normalize statistics of the parent
        dataset.is_normalized_y = self.is_normalized_y
        dataset.y_mean = self.y_mean
        dataset.y_standard_dev = self.y_standard_dev

        # carry over the sub sampling statistics of the parent
        dataset.dataset_min_percentile = self.dataset_min_percentile
        dataset.dataset_max_percentile = self.dataset_max_percentile
        dataset.dataset_min_output = self.dataset_min_output
        dataset.dataset_max_output = self.dataset_max_output
        dataset.dataset_distribution = self.dataset_distribution

        # calculate indices of samples that are visible
        dataset.dataset_visible_mask = visible_mask
        dataset.dataset_size = dataset.y.shape[0]
        return dataset

    def clone(self, subset=None, shard_size=5000,
              to_disk=False, disk_target="dataset", is_absolute=True):
        """Generate a cloned copy of a model-based optimization dataset
        using the provided name and shard generation settings; useful
        when relabelling a dataset buffer from the disk

        Arguments:

        subset: set
            a python set of integers representing the ids of the samples
            to be included in the generated shards
        shard_size: int
            an integer representing the number of samples from a model-based
            optimization data set to save per shard
        to_disk: boolean
            a boolean that indicates whether to store the split data set
            in memory as numpy arrays or to the disk
        disk_target: str
            a string that determines the name and sub folder of the saved
            data set if to_disk is set to be true
        is_absolute: boolean
            a boolean that indicates whether the disk_target path is taken
            relative to the benchmark data folder

        Returns:

        dataset: DatasetBuilder
            an instance of a data set builder subclass containing a copy
            of all data originally associated with this dataset

        """

        # check if the subset is empty
        if subset is not None and len(subset) == 0:
            raise ValueError("cannot pass an empty subset")

        # disable transformations and check the size of the data set
        self._disable_subsample = True
        self._disable_transform = True
        visible_mask = []

        # create lists to store shards and numpy arrays
        partial_shard_x, partial_shard_y = [], []
        x_shards, y_shards = [], []

        # iterate once through the entire data set
        for sample_id, (x, y) in enumerate(self.iterate_samples()):

            # add the sampled x and y to the dataset
            if subset is None or sample_id in subset:
                partial_shard_x.append(x)
                partial_shard_y.append(y)

                # record whether this sample was already visible
                visible_mask.append(self.dataset_visible_mask[sample_id])

            # if the validation shard is large enough then write it
            if (sample_id + 1 == self.dataset_visible_mask.size and
                    len(partial_shard_x) > 0) or \
                    len(partial_shard_x) >= shard_size:

                # stack the sampled x and y values into a shard
                shard_x = np.stack(partial_shard_x, axis=0)
                shard_y = np.stack(partial_shard_y, axis=0)

                if to_disk:

                    # write the design values shard first to a new file
                    x_resource = DiskResource(
                        f"{disk_target}-x-{len(x_shards)}.npy",
                        is_absolute=is_absolute,
                        download_method=None, download_target=None)
                    np.save(x_resource.disk_target, shard_x)
                    shard_x = x_resource

                    # write the prediction values shard second to a new file
                    y_resource = DiskResource(
                        f"{disk_target}-y-{len(y_shards)}.npy",
                        is_absolute=is_absolute,
                        download_method=None, download_target=None)
                    np.save(y_resource.disk_target, shard_y)
                    shard_y = y_resource

                # save the complete shard to a list
                x_shards.append(shard_x)
                y_shards.append(shard_y)

                # clear the buffer of samples for each shard
                partial_shard_x.clear()
                partial_shard_y.clear()

            # at the last sample return two split data sets
            if sample_id + 1 == self.dataset_visible_mask.size:

                # remember to re-enable original transformations
                self._disable_subsample = False
                self._disable_transform = False

                # check if the subset is empty
                if len(x_shards) == 0 or len(y_shards) == 0:
                    raise ValueError("subset produces an empty dataset")

                # return a new version of the dataset
                return self.rebuild_dataset(x_shards, y_shards,
                                            np.stack(visible_mask, axis=0))

    def split(self, val_fraction=0.1, subset=None, shard_size=5000,
              to_disk=False, disk_target="dataset", is_absolute=True):
        """Split a model-based optimization data set into a training set and
        a validation set allocating 'fraction' of the data set to the
        validation set and the rest to the training set

        Arguments:

        val_fraction: float
            a floating point number specifying the fraction of the original
            dataset to split into a validation set
        subset: set
            a python set of integers representing the ids of the samples
            to be included in the generated shards
        shard_size: int
            an integer representing the number of samples from a model-based
            optimization data set to save per shard
        to_disk: boolean
            a boolean that indicates whether to store the split data set
            in memory as numpy arrays or to the disk
        disk_target: str
            a string that determines the name and sub folder of the saved
            data set if to_disk is set to be true
        is_absolute: boolean
            a boolean that indicates whether the disk_target path is taken
            relative to the benchmark data folder

        Returns:

        training_dataset: DatasetBuilder
            an instance of a data set builder subclass containing all data
            points associated with the training set
        validation_dataset: DatasetBuilder
            an instance of a data set builder subclass containing all data
            points associated with the validation set

        """

        # select examples from the active set according to sub sampling
        active_ids = np.where(self.dataset_visible_mask)[0]
        subset = set(active_ids.tolist()) if subset is None else subset
        active_ids = np.array(list(subset))[
            np.random.choice(len(subset), size=int(
                val_fraction * float(len(subset))), replace=False)]

        # generate a set of ids for the validation set
        # noinspection PyTypeChecker
        validation_ids = set(active_ids.tolist())
        training_ids = subset.difference(validation_ids)

        # build a new training  and validation dataset using the split
        # fraction given as an argument
        dtraining = self.clone(subset=training_ids,
                               shard_size=shard_size,
                               to_disk=to_disk,
                               disk_target=f"{disk_target}-train",
                               is_absolute=is_absolute)
        dtraining.freeze_statistics = True

        # intentionally freeze the dataset statistics in order to
        # prevent bugs once a data set is split
        dvalidation = self.clone(subset=validation_ids,
                                 shard_size=shard_size,
                                 to_disk=to_disk,
                                 disk_target=f"{disk_target}-val",
                                 is_absolute=is_absolute)
        dvalidation.freeze_statistics = True

        return dtraining, dvalidation

    def map_normalize_x(self):
        """a function that standardizes the design values 'x' to have zero
        empirical mean and unit empirical variance in the dataset

        """

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are not normalized
        if not self.is_normalized_x:
            self.is_normalized_x = True

        # calculate the normalization statistics in advance
        self.update_x_statistics()

    def map_normalize_y(self):
        """a function that standardizes the prediction values 'y' to have
        zero empirical mean and unit empirical variance in the dataset

        """

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are not normalized
        if not self.is_normalized_y:
            self.is_normalized_y = True

        # calculate the normalization statistics in advance
        self.update_y_statistics()

    def normalize_x(self, x):
        """a function that standardizes the design values 'x' to have
        zero empirical mean and unit empirical variance

        Arguments:

        x: np.ndarray
            a design value represented as a numpy array potentially
            given as a batch of designs which
            shall be normalized according to dataset statistics

        Returns:

        x: np.ndarray
            a design value represented as a numpy array potentially
            given as a batch of designs which
            has been normalized using dataset statistics

        """

        # calculate the mean and standard deviation of the prediction values
        if self.x_mean is None or self.x_standard_dev is None:
            self.update_x_statistics()

        # normalize the prediction values
        return (x - self.x_mean) / self.x_standard_dev

    def normalize_y(self, y):
        """a function that standardizes the prediction values 'y' to have
        zero empirical mean and unit empirical variance

        Arguments:

        y: np.ndarray
            a prediction value represented as a numpy array potentially
            given as a batch of predictions which
            shall be normalized according to dataset statistics

        Returns:

        y: np.ndarray
            a prediction value represented as a numpy array potentially
            given as a batch of predictions which
            has been normalized using dataset statistics

        """

        # calculate the mean and standard deviation of the prediction values
        if self.y_mean is None or self.y_standard_dev is None:
            self.update_y_statistics()

        # normalize the prediction values
        return (y - self.y_mean) / self.y_standard_dev

    def map_denormalize_x(self):
        """a function that un-standardizes the design values 'x' which have
        zero empirical mean and unit empirical variance in the dataset

        """

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are normalized
        if self.is_normalized_x:
            self.is_normalized_x = False

    def map_denormalize_y(self):
        """a function that un-standardizes the prediction values 'y' which
        have zero empirical mean and unit empirical variance in the dataset

        """

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are normalized
        if self.is_normalized_y:
            self.is_normalized_y = False

    def denormalize_x(self, x):
        """a function that un-standardizes the design values 'x' which have
        zero empirical mean and unit empirical variance

        Arguments:

        x: np.ndarray
            a design value represented as a numpy array potentially
            given as a batch of designs which
            shall be denormalized according to dataset statistics

        Returns:

        x: np.ndarray
            a design value represented as a numpy array potentially
            given as a batch of designs which
            has been denormalized using dataset statistics

        """

        # calculate the mean and standard deviation
        if self.x_mean is None or self.x_standard_dev is None:
            self.update_x_statistics()

        # denormalize the prediction values
        return x * self.x_standard_dev + self.x_mean

    def denormalize_y(self, y):
        """a function that un-standardizes the prediction values 'y' which
        have zero empirical mean and unit empirical variance

        Arguments:

        y: np.ndarray
            a prediction value represented as a numpy array potentially
            given as a batch of predictions which
            shall be denormalized according to dataset statistics

        Returns:

        y: np.ndarray
            a prediction value represented as a numpy array potentially
            given as a batch of predictions which
            has been denormalized using dataset statistics

        """

        # calculate the mean and standard deviation
        if self.y_mean is None or self.y_standard_dev is None:
            self.update_y_statistics()

        # denormalize the prediction values
        return y * self.y_standard_dev + self.y_mean
