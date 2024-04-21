from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.disk_resource import DiskResource, SERVER_URL


TF_BIND_8_FILES = ['tf_bind_8-PITX2_T114P_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NR2E3_R76W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NR2E3_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ESX1_K193R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-OVOL2_D228E_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_T333N_R2/tf_bind_8-x-0.npy', 'tf_bind_8-KLF11_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PITX2_R108H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NR2E3_R77Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-SIX6_T165A_R2/tf_bind_8-x-0.npy', 'tf_bind_8-NKX2-8_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-EGR2_E412K_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PROP1_R99Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-WT1_F392L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VAX2_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-CRX_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VSX1_G160D_R1/tf_bind_8-x-0.npy', 'tf_bind_8-WT1_R366L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-CRX_R90W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF655_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NKX2-5_R161P_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF655_S393L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_N47H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-GFI1_L400F_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU4F3_K277R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PHOX2B_R100L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU3F4_A312V_R1/tf_bind_8-x-0.npy', 'tf_bind_8-CRX_E80A_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PITX2_R130W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_F258S_R1/tf_bind_8-x-0.npy', 'tf_bind_8-KLF1_R328H_R2/tf_bind_8-x-0.npy', 'tf_bind_8-VAX2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-MSX2_R172H_R2/tf_bind_8-x-0.npy', 'tf_bind_8-POU6F2_E639K_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF655_E327G_R2/tf_bind_8-x-0.npy', 'tf_bind_8-KLF1_E325K_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_T333N_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PITX2_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-HOXC4_N178S_R1/tf_bind_8-x-0.npy', 'tf_bind_8-MSX2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PHOX2B_R141Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-CRX_R41W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF200_S265Y_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-MSX2_P148H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_R242T_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU4F3_L289F_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_REF_R3/tf_bind_8-x-0.npy', 'tf_bind_8-HESX1_E149K_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_L343Q_R2/tf_bind_8-x-0.npy', 'tf_bind_8-HOXB7_K191R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXB7_K191R_R2/tf_bind_8-x-0.npy', 'tf_bind_8-HESX1_N125S_R2/tf_bind_8-x-0.npy', 'tf_bind_8-BCL6_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NR1H4_C144R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-KLF1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU3F4_R323G_R1/tf_bind_8-x-0.npy', 'tf_bind_8-SNAI2_T234I_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PHOX2B_R141G_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_R192S_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX7_P112L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU3F4_V322M_R1/tf_bind_8-x-0.npy', 'tf_bind_8-EGR2_D383Y_R1/tf_bind_8-x-0.npy', 'tf_bind_8-GFI1B_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-FOXC1_I126M_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_P50L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VAX2_L139M_R2/tf_bind_8-x-0.npy', 'tf_bind_8-HOXB7_M190L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_G48R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HESX1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-CRX_R41Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PBX4_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF655_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-VSX1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXD13_S316C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXD13_I322L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXB7_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-WT1_R394L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU3F4_A237G_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NKX2-5_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ESX1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PROP1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_R128C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXC4_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-FOXC1_S131L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-KLF1_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-NR2E3_G56R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_Y90H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-KLF1_R328H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF200_H322Y_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU4F3_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_P118R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU6F2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ISX_R83Q_R2/tf_bind_8-x-0.npy', 'tf_bind_8-PITX2_L100Q_R2/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PHOX2B_Q143R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-KLF1_H299Y_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_R183C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HESX1_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-HOXD13_R306W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-SIX6_H141N_R2/tf_bind_8-x-0.npy', 'tf_bind_8-FOXC1_F112S_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PITX2_L100Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_R192S_R2/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_R56L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-MSX2_R172H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_R26G_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NKX2-5_T178M_R1/tf_bind_8-x-0.npy', 'tf_bind_8-FOXC1_L130F_R1/tf_bind_8-x-0.npy', 'tf_bind_8-OVOL2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-WT1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PITX2_R108H_R2/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_R332H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NKX2-5_K183E_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU3F4_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_L343Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NKX2-5_R190C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-EGR2_R409W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_P353L_R2/tf_bind_8-x-0.npy', 'tf_bind_8-HOXC4_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-CRX_V66I_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HESX1_R160C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-FOXC1_S82T_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-POU4F3_N316K_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NKX2-8_A94T_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VSX1_R166Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VENTX_E101K_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VAX2_L139M_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_A79E_R1/tf_bind_8-x-0.npy', 'tf_bind_8-FOXC1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-BCL6_H676Y_R1/tf_bind_8-x-0.npy', 'tf_bind_8-GFI1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-SIX6_H141N_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_R270C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXB7_M190L_R2/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF655_S393L_R2/tf_bind_8-x-0.npy', 'tf_bind_8-VSX1_Q175H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_P68S_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXC4_R158L_R2/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_R332H_R2/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_P353R_R2/tf_bind_8-x-0.npy', 'tf_bind_8-FOXC1_P79L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_N47K_R1/tf_bind_8-x-0.npy', 'tf_bind_8-GFI1_N382S_R1/tf_bind_8-x-0.npy', 'tf_bind_8-MSX2_P148H_R2/tf_bind_8-x-0.npy', 'tf_bind_8-KLF11_R402Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-WT1_R394W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-SIX6_T165A_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXB7_R189C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PROP1_R112Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PHOX2B_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PBX4_R215Q_R2/tf_bind_8-x-0.npy', 'tf_bind_8-SIX6_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VENTX_R143C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU6F2_L500M_R1/tf_bind_8-x-0.npy', 'tf_bind_8-SNAI2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXD13_Q325R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXD13_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_P353R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VENTX_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PBX4_R215Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-WT1_H373Y_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_R192H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VSX2_R200Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_V126D_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXC4_R158L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NR1H4_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-GFI1B_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX7_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ISX_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ISX_R83Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-WT1_R366C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-GFI1B_A204T_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_S119R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_P353L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VSX2_R200P_R1/tf_bind_8-x-0.npy', 'tf_bind_8-SNAI2_D119E_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF200_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_R192H_R2/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXB7_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXC4_N178S_R2/tf_bind_8-x-0.npy', 'tf_bind_8-PITX2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-EGR2_R359W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-KLF1_H299Y_R2/tf_bind_8-x-0.npy', 'tf_bind_8-SIX6_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-PBX4_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-ISX_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-GFI1B_A204T_R2/tf_bind_8-x-0.npy', 'tf_bind_8-MSX2_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-EGR2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VSX2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF655_E327G_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HESX1_N125S_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NR2E3_R76Q_R1/tf_bind_8-x-0.npy']


class TFBind8Dataset(DiscreteDataset):
    """A polypeptide synthesis dataset that defines a common set of functions
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

    is_logits: bool (only supported for a DiscreteDataset)
        a value that indicates whether the design values contained in the
        model-based optimization dataset have already been converted to
        logits and need not be converted again

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

    name = "tf_bind_8/tf_bind_8"
    y_name = "enrichment_score"
    x_name = "dna_sequence"

    @staticmethod
    def register_x_shards(transcription_factor='SIX6_REF_R1'):
        """Registers a remote file for download that contains design values
        in a format compatible with the dataset builder class;
        these files are downloaded all at once in the dataset initialization

        Arguments:

        transcription_factor: str
            a string argument that specifies which transcription factor to
            select for model-based optimization, where the goal is to find
            a length 8 polypeptide with maximum binding affinity

        Returns:

        resources: list of RemoteResource
            a list of RemoteResource objects specific to this dataset, which
            will be automatically downloaded while the dataset is built
            and may serve as shards if the dataset is large

        """

        return [DiskResource(
            file, is_absolute=False,
            download_target=file,
            download_method="direct") for file in TF_BIND_8_FILES
            if transcription_factor in file]

    @staticmethod
    def register_y_shards(transcription_factor='SIX6_REF_R1'):
        """Registers a remote file for download that contains prediction
        values in a format compatible with the dataset builder class;
        these files are downloaded all at once in the dataset initialization

        Arguments:

        transcription_factor: str
            a string argument that specifies which transcription factor to
            select for model-based optimization, where the goal is to find
            a length 8 polypeptide with maximum binding affinity

        Returns:

        resources: list of RemoteResource
            a list of RemoteResource objects specific to this dataset, which
            will be automatically downloaded while the dataset is built
            and may serve as shards if the dataset is large

        """

        return [DiskResource(
            file.replace("-x-", "-y-"), is_absolute=False,
            download_target=file.replace('-x-', '-y-'),
            download_method="direct") for file in TF_BIND_8_FILES
            if transcription_factor in file]

    def __init__(self, soft_interpolation=0.6,
                 transcription_factor='SIX6_REF_R1', **kwargs):
        """Initialize a model-based optimization dataset and prepare
        that dataset by loading that dataset from disk and modifying
        its distribution

        Arguments:

        soft_interpolation: float
            a floating point hyper parameter used when converting design values
            from integers to a floating point representation as logits, which
            interpolates between a uniform and dirac distribution
            1.0 = dirac, 0.0 -> uniform
        transcription_factor: str
            a string argument that specifies which transcription factor to
            select for model-based optimization, where the goal is to find
            a length 8 polypeptide with maximum binding affinity
        **kwargs: dict
            additional keyword arguments which are used to parameterize the
            data set generation process, including which shard files are used
            if multiple sets of data set shard files can be loaded

        """

        # set the names the describe the dataset
        self.name = f"tf_bind_8-{transcription_factor}/tf_bind_8"
        self.y_name = "enrichment_score"
        self.x_name = "dna_sequence"

        # initialize the dataset using the method in the base class
        super(TFBind8Dataset, self).__init__(
            self.register_x_shards(transcription_factor=transcription_factor),
            self.register_y_shards(transcription_factor=transcription_factor),
            is_logits=False, num_classes=4,
            soft_interpolation=soft_interpolation, **kwargs)
