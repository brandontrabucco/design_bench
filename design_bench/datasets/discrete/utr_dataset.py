from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.disk_resource import DiskResource


UTR_FILES = ["utr/utr-x-11.npy",
             "utr/utr-x-12.npy",
             "utr/utr-x-3.npy",
             "utr/utr-x-16.npy",
             "utr/utr-x-4.npy",
             "utr/utr-x-5.npy",
             "utr/utr-x-2.npy",
             "utr/utr-x-8.npy",
             "utr/utr-x-15.npy",
             "utr/utr-x-17.npy",
             "utr/utr-x-6.npy",
             "utr/utr-x-21.npy",
             "utr/utr-x-25.npy",
             "utr/utr-x-13.npy",
             "utr/utr-x-14.npy",
             "utr/utr-x-22.npy",
             "utr/utr-x-7.npy",
             "utr/utr-x-20.npy",
             "utr/utr-x-0.npy",
             "utr/utr-x-27.npy",
             "utr/utr-x-26.npy",
             "utr/utr-x-18.npy",
             "utr/utr-x-23.npy",
             "utr/utr-x-10.npy",
             "utr/utr-x-9.npy",
             "utr/utr-x-19.npy",
             "utr/utr-x-24.npy",
             "utr/utr-x-1.npy"]


class UTRDataset(DiscreteDataset):
    """A polypeptide synthesis dataset that defines a common set of functions
    and attributes for a model-based optimization dataset, where the
    goal is to find a design 'x' that maximizes a prediction 'y':

    max_x { y = f(x) }

    Public Attributes:

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

    dataset_max_percentile: float
        the percentile between 0 and 100 of prediction values 'y' above
        which are hidden from access by members outside the class

    dataset_min_percentile: float
        the percentile between 0 and 100 of prediction values 'y' below
        which are hidden from access by members outside the class

    Public Methods:

    subsample(max_percentile: float,
              min_percentile: float):
        a function that exposes a subsampled version of a much larger
        model-based optimization dataset containing design values 'x'
        whose prediction values 'y' are skewed

    relabel(relabel_function:
            Callable[[np.ndarray, np.ndarray], np.ndarray]):
        a function that accepts a function that maps from a dataset of
        design values 'x' and prediction values y to a new set of
        prediction values 'y' and relabels the model-based optimization dataset

    normalize_x(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point design values 'x'
        as input and standardizes them so that they have zero
        empirical mean and unit empirical variance

    denormalize_x(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point design values 'x'
        as input and undoes standardizmoreation so that they have their
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

    --- for discrete tasks only

    to_logits(np.ndarray) > np.ndarray:
        A helper function that accepts design values represented as a numpy
        array of integers as input and converts them to floating point
        logits of a certain probability distribution

    to_integers(np.ndarray) > np.ndarray:
        A helper function that accepts design values represented as a numpy
        array of floating point logits as input and converts them to integer
        representing the max of the distribution

    map_to_logits():
        a function that processes the dataset corresponding to this
        model-based optimization problem, and converts integers to a
        floating point representation as logits

    map_to_integers():
        a function that processes the dataset corresponding to this
        model-based optimization problem, and converts a floating point
        representation as logits to integers

    """

    @staticmethod
    def register_x_shards():
        """Registers a remote file for download that contains design values
        in a format compatible with the dataset builder class;
        these files are downloaded all at once in the dataset initialization

        Returns:

        resources: list of RemoteResource
            a list of RemoteResource objects specific to this dataset, which
            will be automatically downloaded while the dataset is built
            and may serve as shards if the dataset is large

        """

        return [DiskResource(
            file, is_absolute=False,
            download_target=f"https://design-bench."
                            f"s3-us-west-1.amazonaws.com/{file}",
            download_method="direct") for file in UTR_FILES]

    @staticmethod
    def register_y_shards():
        """Registers a remote file for download that contains prediction
        values in a format compatible with the dataset builder class;
        these files are downloaded all at once in the dataset initialization

        Returns:

        resources: list of RemoteResource
            a list of RemoteResource objects specific to this dataset, which
            will be automatically downloaded while the dataset is built
            and may serve as shards if the dataset is large

        """

        return [DiskResource(
            file.replace("-x-", "-y-"), is_absolute=False,
            download_target=f"https://design-bench."
                            f"s3-us-west-1.amazonaws.com/"
                            f"{file.replace('-x-', '-y-')}",
            download_method="direct") for file in UTR_FILES]

    def __init__(self, soft_interpolation=0.6, **kwargs):
        """Initialize a model-based optimization dataset and prepare
        that dataset by loading that dataset from disk and modifying
        its distribution

        Arguments:

        soft_interpolation: float
            a floating point hyper parameter used when converting design values
            from integers to a floating point representation as logits, which
            interpolates between a uniform and dirac distribution
            1.0 = dirac, 0.0 -> uniform
        **kwargs: dict
            additional keyword arguments which are used to parameterize the
            data set generation process, including which shard files are used
            if multiple sets of data set shard files can be loaded

        """

        # initialize the dataset using the method in the base class
        super(UTRDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            is_logits=False, num_classes=4,
            soft_interpolation=soft_interpolation, **kwargs)
