from design_bench.oracles.oracle_builder import OracleBuilder
from design_bench.datasets.dataset_builder import DatasetBuilder
from design_bench.disk_resource import DiskResource, SERVER_URL
from scipy import stats
import numpy as np
import pickle as pkl
import abc
import zipfile


class ApproximateOracle(OracleBuilder, abc.ABC):
    """An abstract class for managing the ground truth score functions f(x)
    for model-based optimization problems, where the
    goal is to find a design 'x' that maximizes a prediction 'y':

    max_x { y = f(x) }

    Public Attributes:

    external_dataset: DatasetBuilder
        an instance of a subclass of the DatasetBuilder class which points to
        the mutable task dataset for a model-based optimization problem

    internal_dataset: DatasetBuilder
        an instance of a subclass of the DatasetBuilder class which has frozen
        statistics and is used for training the oracle

    is_batched: bool
        a boolean variable that indicates whether the evaluation function
        implemented for a particular oracle is batched, which effects
        the scaling coefficient of its computational cost

    internal_batch_size: int
        an integer representing the number of design values to process
        internally at the same time, if None defaults to the entire
        tensor given to the self.score method
    internal_measurements: int
        an integer representing the number of independent measurements of
        the prediction made by the oracle, which are subsequently
        averaged, and is useful when the oracle is stochastic

    noise_std: float
        the standard deviation of gaussian noise added to the prediction
        values 'y' coming out of the ground truth score function f(x)
        in order to make the optimization problem difficult

    expect_normalized_y: bool
        a boolean indicator that specifies whether the inputs to the oracle
        score function are expected to be normalized
    expect_normalized_x: bool
        a boolean indicator that specifies whether the outputs of the oracle
        score function are expected to be normalized
    expect_logits: bool
        a boolean that specifies whether the oracle score function is
        expecting logits when the dataset is discrete

    Public Methods:

    predict(np.ndarray) -> np.ndarray:
        a function that accepts a batch of design values 'x' as input and for
        each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

    check_input_format(DatasetBuilder) -> bool:
        a function that accepts a list of integers as input and returns true
        when design values 'x' with the shape specified by that list are
        compatible with this class of approximate oracle

    fit(np.ndarray, np.ndarray):
        a function that accepts a data set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

    """

    # parameters used for creating a validation set
    default_split_kwargs = dict(val_fraction=0.1, subset=None,
                                shard_size=5000, to_disk=False,
                                disk_target=None, is_absolute=None)

    # parameters used for building the model
    default_model_kwargs = dict()

    @abc.abstractmethod
    def save_model_to_zip(self, model, zip_archive):
        """a function that serializes a machine learning model and stores
        that model in a compressed zip file using the python ZipFile interface
        for sharing and future loading by an ApproximateOracle

        Arguments:

        model: Any
            any format of of machine learning model that will be stored
            in the self.model attribute for later use

        zip_archive: ZipFile
            an instance of the python ZipFile interface that has loaded
            the file path specified by self.resource.disk_target

        """

        raise NotImplementedError

    @abc.abstractmethod
    def load_model_from_zip(self, zip_archive):
        """a function that loads components of a serialized model from a zip
        given zip file using the python ZipFile interface and returns an
        instance of the model

        Arguments:

        zip_archive: ZipFile
            an instance of the python ZipFile interface that has loaded
            the file path specified by self.resource.disk_target

        Returns:

        model: Any
            any format of of machine learning model that will be stored
            in the self.model attribute for later use

        """

        raise NotImplementedError

    @abc.abstractmethod
    def protected_fit(self, training, validation, model_kwargs=None):
        """a function that accepts a training dataset and a validation dataset
        containing design values 'x' and prediction values 'y' in a model-based
        optimization problem and fits an approximate model

        Arguments:

        training: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        validation: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes

        Returns:

        model: Any
            any format of of machine learning model that will be stored
            in the self.model attribute for later use

        """

        raise NotImplementedError

    def fit(self, dataset, split_kwargs=None, model_kwargs=None):
        """a function that accepts a dataset implemented via the DatasetBuilder
        containing design values 'x' and prediction values 'y' in a model-based
        optimization problem and fits an approximate model

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        split_kwargs: dict
            a dictionary of keyword arguments that will be passed to
            dataset.split when constructing a vaidation set
        model_kwargs: dict
            a dictionary of keyword arguments that parameterize the
            architecture and learning algorithm of the model

        Returns:

        model: Any
            any format of of machine learning model that will be stored
            in the self.model attribute for later use

        """

        # load parameters for creating a training and validation set
        final_split_kwargs = self.default_split_kwargs.copy()
        if split_kwargs is not None:
            final_split_kwargs.update(split_kwargs)

        # prepare the dataset for training and validation
        training, validation = dataset.split(**final_split_kwargs)

        # load parameters for fitting a model
        final_model_kwargs = self.default_model_kwargs.copy()
        if model_kwargs is not None:
            final_model_kwargs.update(model_kwargs)

        # fit a specific model to the newly subsampled dataset
        model = self.protected_fit(training, validation,
                                   model_kwargs=final_model_kwargs)

        # evaluate validation rank correlation of model
        rank_correlation = stats.spearmanr(
            self.predict(validation.x, dataset=validation, model=model)[:, 0],
            self.dataset_to_oracle_y(
                validation.y, dataset=validation)[:, 0])[0]

        # return the final model and its training parameters
        return dict(model=model, rank_correlation=rank_correlation,
                    split_kwargs=final_split_kwargs,
                    model_kwargs=final_model_kwargs)

    def __init__(self, dataset: DatasetBuilder,
                 disk_target=None, is_absolute=False, fit=None,
                 split_kwargs=None, model_kwargs=None, **kwargs):
        """Initialize the ground truth score function f(x) for a model-based
        optimization problem, which involves loading the parameters of an
        oracle model and estimating its computational cost

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        disk_target: str
            a path to a zip file that would contain a serialized model, and is
            useful when there are multiple versions of the same model
        is_absolute: bool
            a boolean that indicates whether the provided disk_target path is
            an absolute path or relative to the data folder
        fit: bool
            a boolean that specifies whether the oracle should be re-fit to
            the dataset; only fits the model when not available if None
        split_kwargs: dict
            a dictionary of keyword arguments that will be passed to
            dataset.split when constructing a vaidation set
        model_kwargs: dict
            a dictionary of keyword arguments that parameterize the
            architecture and learning algorithm of the model
        is_batched: bool
            a boolean variable that indicates whether the evaluation function
            implemented for a particular oracle is batched, which effects
            the scaling coefficient of its computational cost
        internal_batch_size: int
            an integer representing the number of design values to process
            internally at the same time, if None defaults to the entire
            tensor given to the self.score method
        internal_measurements: int
            an integer representing the number of independent measurements of
            the prediction made by the oracle, which are subsequently
            averaged, and is useful when the oracle is stochastic
        noise_std: float
            the standard deviation of gaussian noise added to the prediction
            values 'y' coming out of the ground truth score function f(x)
            in order to make the optimization problem difficult
        expect_normalized_y: bool
            a boolean indicator that specifies whether the inputs to the
            oracle score function are expected to be normalized
        expect_normalized_x: bool
            a boolean indicator that specifies whether the outputs of the
            oracle score function are expected to be normalized
        expect_logits: bool
            a boolean that specifies whether the oracle score function
            is expecting logits when the dataset is discrete
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

        # initialize the oracle using the super class
        super(ApproximateOracle, self).__init__(dataset, **kwargs)

        # download the model parameters from s3
        self.resource = self.get_disk_resource(
            dataset, disk_target=disk_target, is_absolute=is_absolute)

        # check if the model is already downloaded
        if (fit is not None and fit) or \
                (not self.resource.is_downloaded
                 and not self.resource.download(unzip=False)):

            # error if not download and cannot fit model
            if fit is not None and not fit:
                raise ValueError("model not downloaded or trained")

            # otherwise build the model
            self.save_params(self.resource.disk_target,
                             self.fit(self.internal_dataset,
                                      split_kwargs=split_kwargs,
                                      model_kwargs=model_kwargs))

        # load the params from disk once its downloaded
        self.params = self.load_params(self.resource.disk_target)

    def get_disk_resource(self, dataset,
                          disk_target=None, is_absolute=False):
        """a function that returns a zip file containing all the files and
        meta data required to re-build an oracle model such as a neural
        network or random forest regression model

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        disk_target: str
            a path to a zip file that would contain a serialized model, and is
            useful when there are multiple versions of the same model
        is_absolute: bool
            a boolean that indicates whether the provided disk_target path is
            an absolute path or relative to the data folder

        Returns:

        model_resource: DiskResource
            a DiskResource instance containing the expected download link of
            the model and the target location on disk where the model is
            expected to be located or downloaded to

        """

        default = f"{dataset.name}/{self.name}.zip"
        return DiskResource(disk_target if disk_target else default,
                            is_absolute=is_absolute,
                            download_method=None if disk_target else "direct",
                            download_target=None if disk_target else
                            default)

    def save_params(self, file, params):
        """a function that serializes a machine learning model and stores
        that model in a compressed zip file using the python ZipFile interface
        for sharing and future loading by an ApproximateOracle

        Arguments:

        file: str
            a path to a zip file that would contain a serialized model, and is
            useful when there are multiple versions of the same model
        params: Any
            a dictionary of parameters containing a machine learning model
            and values that describe its architecture and performance

        """

        # open a zip archive that will contain a model
        with zipfile.ZipFile(file, mode="w") as zip_archive:
            self.save_model_to_zip(params["model"], zip_archive)

            # write the validation rank correlation to the zip file
            with zip_archive.open('rank_correlation.npy', "w") as file:
                file.write(params["rank_correlation"].dumps())

            # write the validation parameters to the zip file
            with zip_archive.open('split_kwargs.pkl', "w") as file:
                file.write(pkl.dumps(params["split_kwargs"]))

            # write the model parameters to the zip file
            with zip_archive.open('model_kwargs.pkl', "w") as file:
                file.write(pkl.dumps(params["model_kwargs"]))

    def load_params(self, file):
        """a function that loads components of a serialized model from a zip
        given zip file using the python ZipFile interface and returns an
        instance of the model

        Arguments:

        file: str
            a path to a zip file that would contain a serialized model, and is
            useful when there are multiple versions of the same model

        Returns:

        params: Any
            a dictionary of parameters containing a machine learning model
            and values that describe its architecture and performance

        """

        # open a zip archive that contains a model
        with zipfile.ZipFile(file, mode="r") as zip_archive:
            model = self.load_model_from_zip(zip_archive)

            # read the validation rank correlation from the zip file
            with zip_archive.open('rank_correlation.npy', "r") as file:
                rank_correlation = pkl.loads(file.read())

            # read the validation parameters from the zip file
            with zip_archive.open('split_kwargs.pkl', "r") as file:
                split_kwargs = pkl.loads(file.read())

            # read the model parameters from the zip file
            with zip_archive.open('model_kwargs.pkl', "r") as file:
                model_kwargs = pkl.loads(file.read())

        # return the final model and its training parameters
        return dict(model=model, rank_correlation=rank_correlation,
                    split_kwargs=split_kwargs, model_kwargs=model_kwargs)
