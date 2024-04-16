from design_bench.oracles.feature_extractors.feature_extractor import FeatureExtractor
from design_bench.disk_resource import DiskResource
from design_bench.disk_resource import DATA_DIR
from design_bench.disk_resource import SERVER_URL
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
import transformers
import deepchem.feat as feat
import os
import numpy as np

# To avoid the following warning spam:
# "Setting 'max_len_single_sentence' is now deprecated. This value is automatically set up."
# "Setting 'max_len_sentences_pair' is now deprecated. This value is automatically set up."
transformers.logging.set_verbosity_error()

class MorganFingerprintFeatures(FeatureExtractor):
    """An abstract class for managing transformations applied to model-based
    optimization datasets when constructing the oracle; for example, if the
    oracle is intended to learn from molecule fingerprints

    max_x { y = f(x) }

    Public Methods:

    dataset_to_oracle_x(np.ndarray) -> np.ndarray
        Helper function for converting from designs contained in the
        dataset format into a format the oracle is expecting to process,
        such as from integers to logits of a categorical distribution

    dataset_to_oracle_y(np.ndarray) -> np.ndarray
        Helper function for converting from predictions contained in the
        dataset format into a format the oracle is expecting to process,
        such as from normalized to denormalized predictions

    oracle_to_dataset_x(np.ndarray) -> np.ndarray
        Helper function for converting from designs in the format of the
        oracle into the design format the dataset contains, such as
        from categorical logits to integers

    oracle_to_dataset_y(np.ndarray) -> np.ndarray
        Helper function for converting from predictions in the
        format of the oracle into a format the dataset contains,
        such as from normalized to denormalized predictions

    """

    name = "morgan_fingerprint"

    def __init__(self, size=2048, radius=4, dtype=np.int32):
        """An abstract class for managing transformations applied to
        model-based optimization datasets when constructing the oracle; for
        example, if the oracle learns from molecule fingerprints

        Arguments:

        size: int
            the number of bits in the morgan fingerprint returned by RDKit,
            controls the vector size of the molecule embedding
        radius: int
            the substructure radius passed to RDKit that controls how local
            the information encoded in the molecule embedding is

        """

        # wrap the deepchem featurizer that relies on rdkit
        self.featurizer = feat.CircularFingerprint(size=size, radius=radius)
        self.size = size
        self.radius = radius
        self.dtype = dtype

        # download the vocabulary file is not present in the cache
        vocab_file = DiskResource(
            os.path.join(DATA_DIR, 'smiles_vocab.txt'),
            download_method="direct",
            download_target=f'{SERVER_URL}/smiles_vocab.txt')
        if not vocab_file.is_downloaded:
            vocab_file.download()
        self.tokenizer = SmilesTokenizer(
            os.path.join(DATA_DIR, 'smiles_vocab.txt'))

    def dataset_to_oracle_x(self, x_batch, dataset):
        """Helper function for converting from designs contained in the
        dataset format into a format the oracle is expecting to process,
        such as from integers to logits of a categorical distribution

        Arguments:

        x_batch: np.ndarray
            a batch of design values 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned
        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        x_batch: np.ndarray
            a batch of design values 'x' that have been converted from the
            format of designs contained in the dataset to the format
            expected by the oracle score function

        """

        x_out = []
        for xi in x_batch:

            # identify stop and start tokens so they can be removed
            stop_tokens = np.where(xi == 13)[0]
            tokens = xi[1:stop_tokens[0]] if stop_tokens.size > 0 else xi[1:]

            # apply morgan fingerprint featurization using rdkit
            token_string = self.tokenizer.decode(tokens).replace(" ", "")
            value = self.featurizer.featurize(token_string)[0]

            # collate all results into a single numpy array
            x_out.append(np.full([self.size], np.NaN, dtype=self.dtype)
                         if value.size != self.size
                         else np.array(value, dtype=self.dtype))

        return np.asarray(x_out)

    def dataset_to_oracle_y(self, y_batch, dataset):
        """Helper function for converting from predictions contained in the
        dataset format into a format the oracle is expecting to process,
        such as from normalized to denormalized predictions

        Arguments:

        y_batch: np.ndarray
            a batch of prediction values 'y' that are from the dataset and
            will be processed into a format expected by the oracle score
            function, which is useful when training the oracle
        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        y_batch: np.ndarray
            a batch of prediction values 'y' that have been converted from
            the format of predictions contained in the dataset to the
            format expected by the oracle score function

        """

        return y_batch

    def oracle_to_dataset_x(self, x_batch, dataset):
        """Helper function for converting from designs in the format of the
        oracle into the design format the dataset contains, such as
        from categorical logits to integers

        Arguments:

        x_batch: np.ndarray
            a batch of design values 'x' that have been converted from
            the format of designs contained in the dataset to the
            format expected by the oracle score function
        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        x_batch: np.ndarray
            a batch of design values 'x' that have been converted from
            the format of the oracle to the format of designs
            contained in the dataset

        """

        raise NotImplementedError("features are not invertible")

    def oracle_to_dataset_y(self, y_batch, dataset):
        """Helper function for converting from predictions in the format
        of the oracle into a format the dataset contains, such as
        from normalized to denormalized predictions

        Arguments:

        y_batch: np.ndarray
            a batch of prediction values 'y' that have been converted from
            the format of predictions contained in the dataset to the
            format expected by the oracle score function
        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        y_batch: np.ndarray
            a batch of prediction values 'y' that have been converted from
            the format of the oracle to the format of predictions
            contained in the dataset

        """

        return y_batch

    def input_shape(self, dataset):
        """Helper function for converting from predictions in the format
        of the oracle into a format the dataset contains, such as
        from normalized to denormalized predictions

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        input_shape: List[int]
            the shape of input tensors that were sampled from the dataset and
            are transformed into features using subclasses of this class

        """

        return [self.size]

    def input_dtype(self, dataset):
        """Helper function that returns the data type of the features returned
        by running the feature extractor from dataset to oracle

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        input_dtype: List[int]
            the type of input tensors that were sampled from the dataset and
            are transformed into features using subclasses of this class

        """

        return self.dtype

    def is_discrete(self, dataset):
        """Helper function that specifies whether the transformation applied
        by the feature extractor returns a discrete or continuous set of
        features, which is required for building predictive models

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        is_discrete: bool
            a boolean that indicates whether the dataset has been transformed
            into a discrete or continuous representation

        """

        return np.issubdtype(self.dtype, np.integer)

    def num_classes(self, dataset):
        """Helper function for determining the number of classes in the discrete
        representation intended for the oracle, if it is discrete, otherwise
        this function may not be implemented and will raise an error

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        num_classes: int
            the number of classes in the discrete representation for the model
            based optimization dataset used for training the oracle

        """

        if not self.is_discrete(dataset):
            raise NotImplementedError("continuous features do not have ids")
        return 2
