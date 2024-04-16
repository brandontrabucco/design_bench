from design_bench.registration import registry, register, make, spec
from design_bench.oracles.sklearn.kernels import ProteinKernel
from design_bench.oracles.sklearn.kernels import DefaultSequenceKernel
from design_bench.oracles.feature_extractors.\
    morgan_fingerprint_features import MorganFingerprintFeatures
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import os
import numpy as np
from huggingface_hub import snapshot_download 

DB_HF_DATASET = os.environ.get("DB_HF_REPO", "beckhamc/design_bench_data")
DB_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "design_bench_data"
)
snapshot_download(DB_HF_DATASET, repo_type="dataset", local_dir=DB_DATA_DIR)

register('ToyDiscrete-Exact-v0',
         'design_bench.datasets.discrete.toy_discrete_dataset:ToyDiscreteDataset',
         'design_bench.oracles.exact:ToyDiscreteOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=40,
             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('GFP-GP-v0',
         'design_bench.datasets.discrete.gfp_dataset:GFPDataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=5000,
             distribution=None,
             max_percentile=60,
             min_percentile=50),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(kernel=ProteinKernel()),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="gfp/split",
                               is_absolute=False)))


register('GFP-RandomForest-v0',
         'design_bench.datasets.discrete.gfp_dataset:GFPDataset',
         'design_bench.oracles.sklearn:RandomForestOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=5000,
             distribution=None,
             max_percentile=60,
             min_percentile=50),

         # keyword arguments for building RandomForest oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(n_estimators=100,
                               max_depth=100,
                               max_features="auto"),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="gfp/split",
                               is_absolute=False)))


register('GFP-FullyConnected-v0',
         'design_bench.datasets.discrete.gfp_dataset:GFPDataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=5000,
             distribution=None,
             max_percentile=60,
             min_percentile=50),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(embedding_size=64,
                               hidden_size=512,
                               activation='relu',
                               num_layers=2,
                               epochs=5,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="gfp/split",
                               is_absolute=False)))


register('GFP-LSTM-v0',
         'design_bench.datasets.discrete.gfp_dataset:GFPDataset',
         'design_bench.oracles.tensorflow:LSTMOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=5000,
             distribution=None,
             max_percentile=60,
             min_percentile=50),

         # keyword arguments for training LSTM oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               num_layers=2,
                               epochs=50,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="gfp/split",
                               is_absolute=False)))


register('GFP-ResNet-v0',
         'design_bench.datasets.discrete.gfp_dataset:GFPDataset',
         'design_bench.oracles.tensorflow:ResNetOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=5000,
             distribution=None,
             max_percentile=60,
             min_percentile=50),

         # keyword arguments for training ResNet oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               activation='relu',
                               kernel_size=3,
                               num_blocks=4,
                               epochs=50,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="gfp/split",
                               is_absolute=False)))


register('GFP-Transformer-v0',
         'design_bench.datasets.discrete.gfp_dataset:GFPDataset',
         'design_bench.oracles.tensorflow:TransformerOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=5000,
             distribution=None,
             max_percentile=60,
             min_percentile=50),

         # keyword arguments for training Transformer oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             internal_batch_size=32,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               feed_forward_size=256,
                               activation='relu',
                               num_heads=2,
                               num_blocks=4,
                               epochs=20,
                               shuffle_buffer=60000,
                               learning_rate=0.0001,
                               dropout_rate=0.1),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="gfp/split",
                               is_absolute=False)))


register('TFBind8-Exact-v0',
         'design_bench.datasets.discrete.tf_bind_8_dataset:TFBind8Dataset',
         'design_bench.oracles.exact:TFBind8Oracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=50,
             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('TFBind8-GP-v0',
         'design_bench.datasets.discrete.tf_bind_8_dataset:TFBind8Dataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=50,
             min_percentile=0,
             transcription_factor="SIX6_REF_R1"),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(kernel=DefaultSequenceKernel(size=4)),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="tf_bind_8-SIX6_REF_R1/split",
                               is_absolute=False)))


register('TFBind8-RandomForest-v0',
         'design_bench.datasets.discrete.tf_bind_8_dataset:TFBind8Dataset',
         'design_bench.oracles.sklearn:RandomForestOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=50,
             min_percentile=0,
             transcription_factor="SIX6_REF_R1"),

         # keyword arguments for building RandomForest oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(n_estimators=100,
                               max_depth=100,
                               max_features="auto"),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="tf_bind_8-SIX6_REF_R1/split",
                               is_absolute=False)))


register('TFBind8-FullyConnected-v0',
         'design_bench.datasets.discrete.tf_bind_8_dataset:TFBind8Dataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=50,
             min_percentile=0,
             transcription_factor="SIX6_REF_R1"),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(embedding_size=64,
                               hidden_size=512,
                               activation='relu',
                               num_layers=2,
                               epochs=5,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="tf_bind_8-SIX6_REF_R1/split",
                               is_absolute=False)))


register('TFBind8-LSTM-v0',
         'design_bench.datasets.discrete.tf_bind_8_dataset:TFBind8Dataset',
         'design_bench.oracles.tensorflow:LSTMOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=50,
             min_percentile=0,
             transcription_factor="SIX6_REF_R1"),

         # keyword arguments for training LSTM oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               num_layers=2,
                               epochs=50,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.01,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="tf_bind_8-SIX6_REF_R1/split",
                               is_absolute=False)))


register('TFBind8-ResNet-v0',
         'design_bench.datasets.discrete.tf_bind_8_dataset:TFBind8Dataset',
         'design_bench.oracles.tensorflow:ResNetOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=50,
             min_percentile=0,
             transcription_factor="SIX6_REF_R1"),

         # keyword arguments for training ResNet oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               activation='relu',
                               kernel_size=3,
                               num_blocks=4,
                               epochs=50,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="tf_bind_8-SIX6_REF_R1/split",
                               is_absolute=False)))


register('TFBind8-Transformer-v0',
         'design_bench.datasets.discrete.tf_bind_8_dataset:TFBind8Dataset',
         'design_bench.oracles.tensorflow:TransformerOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=50,
             min_percentile=0,
             transcription_factor="SIX6_REF_R1"),

         # keyword arguments for training Transformer oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             internal_batch_size=32,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               feed_forward_size=256,
                               activation='relu',
                               num_heads=2,
                               num_blocks=4,
                               epochs=50,
                               shuffle_buffer=5000,
                               learning_rate=0.0001,
                               dropout_rate=0.1),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="tf_bind_8-SIX6_REF_R1/split",
                               is_absolute=False)))


register('TFBind10-Exact-v0',
         'design_bench.datasets.discrete.tf_bind_10_dataset:TFBind10Dataset',
         'design_bench.oracles.exact.tf_bind_10_oracle:TFBind10Oracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=50,
             min_percentile=0,
             transcription_factor='pho4'),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('CIFARNAS-Exact-v0',
         'design_bench.datasets.discrete.cifar_nas_dataset:CIFARNASDataset',
         'design_bench.oracles.exact.cifar_nas_oracle:CIFARNASOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=50,
             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('NASBench-Exact-v0',
         'design_bench.datasets.discrete.nas_bench_dataset:NASBenchDataset',
         'design_bench.oracles.exact:NASBenchOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=30,
             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('UTR-GP-v0',
         'design_bench.datasets.discrete.utr_dataset:UTRDataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=50,
             min_percentile=0),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=5000,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(kernel=DefaultSequenceKernel(size=4)),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.2,
                               subset=None,
                               shard_size=5000,
                               to_disk=False,
                               disk_target=None,
                               is_absolute=None)))


register('UTR-RandomForest-v0',
         'design_bench.datasets.discrete.utr_dataset:UTRDataset',
         'design_bench.oracles.sklearn:RandomForestOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=50,
             min_percentile=0),

         # keyword arguments for building RandomForest oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=5000,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(n_estimators=100,
                               max_depth=100,
                               max_features="auto"),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.2,
                               subset=None,
                               shard_size=5000,
                               to_disk=False,
                               disk_target=None,
                               is_absolute=None)))


register('UTR-FullyConnected-v0',
         'design_bench.datasets.discrete.utr_dataset:UTRDataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=50,
             min_percentile=0),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(embedding_size=64,
                               hidden_size=512,
                               activation='relu',
                               num_layers=2,
                               epochs=5,
                               shuffle_buffer=260000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.07142857142,
                               subset=None,
                               shard_size=280000,
                               to_disk=False,
                               disk_target=None,
                               is_absolute=None)))


register('UTR-LSTM-v0',
         'design_bench.datasets.discrete.utr_dataset:UTRDataset',
         'design_bench.oracles.tensorflow:LSTMOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=50,
             min_percentile=0),

         # keyword arguments for training LSTM oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=64,
                               num_layers=2,
                               epochs=5,
                               shuffle_buffer=260000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.07142857142,
                               subset=None,
                               shard_size=280000,
                               to_disk=False,
                               disk_target=None,
                               is_absolute=None)))


register('UTR-ResNet-v0',
         'design_bench.datasets.discrete.utr_dataset:UTRDataset',
         'design_bench.oracles.tensorflow:ResNetOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=50,
             min_percentile=0),

         # keyword arguments for training ResNet oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             internal_batch_size=128,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=120,
                               activation='relu',
                               kernel_size=8,
                               num_blocks=2,
                               epochs=5,
                               shuffle_buffer=260000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.07142857142,
                               subset=None,
                               shard_size=280000,
                               to_disk=False,
                               disk_target=None,
                               is_absolute=None)))


register('UTR-Transformer-v0',
         'design_bench.datasets.discrete.utr_dataset:UTRDataset',
         'design_bench.oracles.tensorflow:TransformerOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=50,
             min_percentile=0),

         # keyword arguments for training Transformer oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             internal_batch_size=128,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=256,
                               feed_forward_size=256,
                               activation='relu',
                               num_heads=4,
                               num_blocks=2,
                               epochs=5,
                               shuffle_buffer=260000,
                               learning_rate=0.001,
                               dropout_rate=0.1),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.07142857142,
                               subset=None,
                               shard_size=280000,
                               to_disk=False,
                               disk_target=None,
                               is_absolute=None)))


for standard_type, assay_chembl_id in [('MCHC', 'CHEMBL3885882'),
                                       ('CHLORIDE', 'CHEMBL3885882'),
                                       ('MCH', 'CHEMBL3885882'),
                                       ('CREAT', 'CHEMBL3885882'),
                                       ('PHOS', 'CHEMBL3885882'),
                                       ('SODIUM', 'CHEMBL3885882')]:

    register(f'ChEMBL_{standard_type}_{assay_chembl_id}_MorganFingerprint-GP-v0',
             'design_bench.datasets.discrete.chembl_dataset:ChEMBLDataset',
             'design_bench.oracles.sklearn:GaussianProcessOracle',

             # keyword arguments for building the dataset
             dataset_kwargs=dict(
                 max_samples=None,
                 distribution=None,
                 max_percentile=40,
                 min_percentile=0,
                 assay_chembl_id=assay_chembl_id,
                 standard_type=standard_type),

             # keyword arguments for building GP oracle
             oracle_kwargs=dict(
                 noise_std=0.0,
                 max_samples=2000,
                 distribution=None,
                 max_percentile=100,
                 min_percentile=0,

                 # process the data into morgan fingerprints
                 feature_extractor=MorganFingerprintFeatures(dtype=np.int32),

                 # parameters used for building the model
                 model_kwargs=dict(kernel=DefaultSequenceKernel(size=2),
                                   alpha=0.01),

                 # parameters used for building the validation set
                 split_kwargs=dict(val_fraction=0.5,
                                   subset=None,
                                   shard_size=50000,
                                   to_disk=False)))

    register(f'ChEMBL_{standard_type}_{assay_chembl_id}-GP-v0',
             'design_bench.datasets.discrete.chembl_dataset:ChEMBLDataset',
             'design_bench.oracles.sklearn:GaussianProcessOracle',

             # keyword arguments for building the dataset
             dataset_kwargs=dict(
                 max_samples=None,
                 distribution=None,
                 max_percentile=40,
                 min_percentile=0,
                 assay_chembl_id=assay_chembl_id,
                 standard_type=standard_type),

             # keyword arguments for building GP oracle
             oracle_kwargs=dict(
                 noise_std=0.0,
                 max_samples=2000,
                 distribution=None,
                 max_percentile=100,
                 min_percentile=0,

                 # parameters used for building the model
                 model_kwargs=dict(kernel=DefaultSequenceKernel(size=591),
                                   alpha=0.01),

                 # parameters used for building the validation set
                 split_kwargs=dict(val_fraction=0.5,
                                   subset=None,
                                   shard_size=50000,
                                   to_disk=False)))

    register(f'ChEMBL_{standard_type}_{assay_chembl_id}_MorganFingerprint-RandomForest-v0',
             'design_bench.datasets.discrete.chembl_dataset:ChEMBLDataset',
             'design_bench.oracles.sklearn:RandomForestOracle',

             # keyword arguments for building the dataset
             dataset_kwargs=dict(
                 max_samples=None,
                 distribution=None,
                 max_percentile=40,
                 min_percentile=0,
                 assay_chembl_id=assay_chembl_id,
                 standard_type=standard_type),

             # keyword arguments for building RandomForest oracle
             oracle_kwargs=dict(
                 noise_std=0.0,
                 max_samples=2000,
                 distribution=None,
                 max_percentile=100,
                 min_percentile=0,

                 # process the data into morgan fingerprints
                 override_input_spec=True,
                 feature_extractor=MorganFingerprintFeatures(dtype=np.float32),

                 # parameters used for building the model
                 model_kwargs=dict(n_estimators=100,
                                   max_depth=100,
                                   max_features="auto"),

                 # parameters used for building the validation set
                 split_kwargs=dict(val_fraction=0.5,
                                   subset=None,
                                   shard_size=50000,
                                   to_disk=False)))

    register(f'ChEMBL_{standard_type}_{assay_chembl_id}-RandomForest-v0',
             'design_bench.datasets.discrete.chembl_dataset:ChEMBLDataset',
             'design_bench.oracles.sklearn:RandomForestOracle',

             # keyword arguments for building the dataset
             dataset_kwargs=dict(
                 max_samples=None,
                 distribution=None,
                 max_percentile=40,
                 min_percentile=0,
                 assay_chembl_id=assay_chembl_id,
                 standard_type=standard_type),

             # keyword arguments for building RandomForest oracle
             oracle_kwargs=dict(
                 noise_std=0.0,
                 max_samples=2000,
                 distribution=None,
                 max_percentile=100,
                 min_percentile=0,

                 # parameters used for building the model
                 model_kwargs=dict(n_estimators=100,
                                   max_depth=100,
                                   max_features="auto"),

                 # parameters used for building the validation set
                 split_kwargs=dict(val_fraction=0.5,
                                   subset=None,
                                   shard_size=50000,
                                   to_disk=False)))

    register(f'ChEMBL_{standard_type}_{assay_chembl_id}_MorganFingerprint-FullyConnected-v0',
             'design_bench.datasets.discrete.chembl_dataset:ChEMBLDataset',
             'design_bench.oracles.tensorflow:FullyConnectedOracle',

             # keyword arguments for building the dataset
             dataset_kwargs=dict(
                 max_samples=None,
                 distribution=None,
                 max_percentile=40,
                 min_percentile=0,
                 assay_chembl_id=assay_chembl_id,
                 standard_type=standard_type),

             # keyword arguments for training FullyConnected oracle
             oracle_kwargs=dict(
                 noise_std=0.0,
                 max_samples=None,
                 distribution=None,
                 max_percentile=100,
                 min_percentile=0,

                 # process the data into morgan fingerprints
                 feature_extractor=MorganFingerprintFeatures(dtype=np.float32),

                 # parameters used for building the model
                 model_kwargs=dict(embedding_size=32,
                                   hidden_size=512,
                                   activation='relu',
                                   num_layers=2,
                                   epochs=5,
                                   shuffle_buffer=5000,
                                   learning_rate=0.0001),

                 # parameters used for building the validation set
                 split_kwargs=dict(val_fraction=0.1,
                                   subset=None,
                                   shard_size=50000,
                                   to_disk=False)))

    register(f'ChEMBL_{standard_type}_{assay_chembl_id}-FullyConnected-v0',
             'design_bench.datasets.discrete.chembl_dataset:ChEMBLDataset',
             'design_bench.oracles.tensorflow:FullyConnectedOracle',

             # keyword arguments for building the dataset
             dataset_kwargs=dict(
                 max_samples=None,
                 distribution=None,
                 max_percentile=40,
                 min_percentile=0,
                 assay_chembl_id=assay_chembl_id,
                 standard_type=standard_type),

             # keyword arguments for training FullyConnected oracle
             oracle_kwargs=dict(
                 noise_std=0.0,
                 max_samples=None,
                 distribution=None,
                 max_percentile=100,
                 min_percentile=0,

                 # parameters used for building the model
                 model_kwargs=dict(embedding_size=32,
                                   hidden_size=512,
                                   activation='relu',
                                   num_layers=2,
                                   epochs=20,
                                   shuffle_buffer=5000,
                                   learning_rate=0.0001),

                 # parameters used for building the validation set
                 split_kwargs=dict(val_fraction=0.1,
                                   subset=None,
                                   shard_size=50000,
                                   to_disk=False)))

    register(f'ChEMBL_{standard_type}_{assay_chembl_id}_MorganFingerprint-LSTM-v0',
             'design_bench.datasets.discrete.chembl_dataset:ChEMBLDataset',
             'design_bench.oracles.tensorflow:LSTMOracle',

             # keyword arguments for building the dataset
             dataset_kwargs=dict(
                 max_samples=None,
                 distribution=None,
                 max_percentile=40,
                 min_percentile=0,
                 assay_chembl_id=assay_chembl_id,
                 standard_type=standard_type),

             # keyword arguments for training LSTM oracle
             oracle_kwargs=dict(
                 noise_std=0.0,
                 max_samples=None,
                 distribution=None,
                 max_percentile=100,
                 min_percentile=0,

                 # process the data into morgan fingerprints
                 feature_extractor=MorganFingerprintFeatures(dtype=np.int32),

                 # parameters used for building the model
                 model_kwargs=dict(hidden_size=64,
                                   num_layers=2,
                                   epochs=20,
                                   shuffle_buffer=5000,
                                   learning_rate=0.001),

                 # parameters used for building the validation set
                 split_kwargs=dict(val_fraction=0.1,
                                   subset=None,
                                   shard_size=50000,
                                   to_disk=False)))

    register(f'ChEMBL_{standard_type}_{assay_chembl_id}-LSTM-v0',
             'design_bench.datasets.discrete.chembl_dataset:ChEMBLDataset',
             'design_bench.oracles.tensorflow:LSTMOracle',

             # keyword arguments for building the dataset
             dataset_kwargs=dict(
                 max_samples=None,
                 distribution=None,
                 max_percentile=40,
                 min_percentile=0,
                 assay_chembl_id=assay_chembl_id,
                 standard_type=standard_type),

             # keyword arguments for training LSTM oracle
             oracle_kwargs=dict(
                 noise_std=0.0,
                 max_samples=None,
                 distribution=None,
                 max_percentile=100,
                 min_percentile=0,

                 # parameters used for building the model
                 model_kwargs=dict(hidden_size=64,
                                   num_layers=2,
                                   epochs=20,
                                   shuffle_buffer=5000,
                                   learning_rate=0.001),

                 # parameters used for building the validation set
                 split_kwargs=dict(val_fraction=0.1,
                                   subset=None,
                                   shard_size=50000,
                                   to_disk=False)))

    register(f'ChEMBL_{standard_type}_{assay_chembl_id}_MorganFingerprint-ResNet-v0',
             'design_bench.datasets.discrete.chembl_dataset:ChEMBLDataset',
             'design_bench.oracles.tensorflow:ResNetOracle',

             # keyword arguments for building the dataset
             dataset_kwargs=dict(
                 max_samples=None,
                 distribution=None,
                 max_percentile=40,
                 min_percentile=0,
                 assay_chembl_id=assay_chembl_id,
                 standard_type=standard_type),

             # keyword arguments for training ResNet oracle
             oracle_kwargs=dict(
                 noise_std=0.0,
                 max_samples=None,
                 distribution=None,
                 max_percentile=100,
                 min_percentile=0,

                 # process the data into morgan fingerprints
                 feature_extractor=MorganFingerprintFeatures(dtype=np.int32),

                 # parameters used for building the model
                 model_kwargs=dict(hidden_size=64,
                                   activation='relu',
                                   kernel_size=3,
                                   num_blocks=4,
                                   epochs=20,
                                   shuffle_buffer=5000,
                                   learning_rate=0.001),

                 # parameters used for building the validation set
                 split_kwargs=dict(val_fraction=0.1,
                                   subset=None,
                                   shard_size=50000,
                                   to_disk=False)))

    register(f'ChEMBL_{standard_type}_{assay_chembl_id}-ResNet-v0',
             'design_bench.datasets.discrete.chembl_dataset:ChEMBLDataset',
             'design_bench.oracles.tensorflow:ResNetOracle',

             # keyword arguments for building the dataset
             dataset_kwargs=dict(
                 max_samples=None,
                 distribution=None,
                 max_percentile=40,
                 min_percentile=0,
                 assay_chembl_id=assay_chembl_id,
                 standard_type=standard_type),

             # keyword arguments for training ResNet oracle
             oracle_kwargs=dict(
                 noise_std=0.0,
                 max_samples=None,
                 distribution=None,
                 max_percentile=100,
                 min_percentile=0,

                 # parameters used for building the model
                 model_kwargs=dict(hidden_size=64,
                                   activation='relu',
                                   kernel_size=3,
                                   num_blocks=4,
                                   epochs=20,
                                   shuffle_buffer=5000,
                                   learning_rate=0.001),

                 # parameters used for building the validation set
                 split_kwargs=dict(val_fraction=0.1,
                                   subset=None,
                                   shard_size=50000,
                                   to_disk=False)))

    register(f'ChEMBL_{standard_type}_{assay_chembl_id}_MorganFingerprint-Transformer-v0',
             'design_bench.datasets.discrete.chembl_dataset:ChEMBLDataset',
             'design_bench.oracles.tensorflow:TransformerOracle',

             # keyword arguments for building the dataset
             dataset_kwargs=dict(
                 max_samples=None,
                 distribution=None,
                 max_percentile=40,
                 min_percentile=0,
                 assay_chembl_id=assay_chembl_id,
                 standard_type=standard_type),

             # keyword arguments for training Transformer oracle
             oracle_kwargs=dict(
                 noise_std=0.0,
                 internal_batch_size=32,
                 max_samples=None,
                 distribution=None,
                 max_percentile=100,
                 min_percentile=0,

                 # process the data into morgan fingerprints
                 feature_extractor=MorganFingerprintFeatures(dtype=np.int32),

                 # parameters used for building the model
                 model_kwargs=dict(hidden_size=128,
                                   feed_forward_size=512,
                                   activation='relu',
                                   num_heads=4,
                                   num_blocks=4,
                                   epochs=20,
                                   shuffle_buffer=20000,
                                   learning_rate=0.0001,
                                   dropout_rate=0.2),

                 # parameters used for building the validation set
                 split_kwargs=dict(val_fraction=0.1,
                                   subset=None,
                                   shard_size=50000,
                                   to_disk=False)))

    register(f'ChEMBL_{standard_type}_{assay_chembl_id}-Transformer-v0',
             'design_bench.datasets.discrete.chembl_dataset:ChEMBLDataset',
             'design_bench.oracles.tensorflow:TransformerOracle',

             # keyword arguments for building the dataset
             dataset_kwargs=dict(
                 max_samples=None,
                 distribution=None,
                 max_percentile=40,
                 min_percentile=0,
                 assay_chembl_id=assay_chembl_id,
                 standard_type=standard_type),

             # keyword arguments for training Transformer oracle
             oracle_kwargs=dict(
                 noise_std=0.0,
                 internal_batch_size=32,
                 max_samples=None,
                 distribution=None,
                 max_percentile=100,
                 min_percentile=0,

                 # parameters used for building the model
                 model_kwargs=dict(hidden_size=128,
                                   feed_forward_size=512,
                                   activation='relu',
                                   num_heads=4,
                                   num_blocks=4,
                                   epochs=20,
                                   shuffle_buffer=20000,
                                   learning_rate=0.0001,
                                   dropout_rate=0.2),

                 # parameters used for building the validation set
                 split_kwargs=dict(val_fraction=0.1,
                                   subset=None,
                                   shard_size=50000,
                                   to_disk=False)))


register('ToyContinuous-Exact-v0',
         'design_bench.datasets.continuous.toy_continuous_dataset:ToyContinuousDataset',
         'design_bench.oracles.exact:ToyContinuousOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=40,
             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('HopperController-Exact-v0',
         'design_bench.datasets.continuous.hopper_controller_dataset:HopperControllerDataset',
         'design_bench.oracles.exact:HopperControllerOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('HopperController-Exact-v1',
         'design_bench.datasets.continuous.hopper_controller_dataset:HopperControllerDataset',
         'design_bench.oracles.exact:HopperControllerStochasticOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('HopperController-GP-v0',
         'design_bench.datasets.continuous.hopper_controller_dataset:HopperControllerDataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(kernel=ConstantKernel(
                 constant_value=1.0, constant_value_bounds=(0.0, 10.0)) *
                 RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) +
                 RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="hopper_controller/split",
                               is_absolute=False)))


register('HopperController-RandomForest-v0',
         'design_bench.datasets.continuous.hopper_controller_dataset:HopperControllerDataset',
         'design_bench.oracles.sklearn:RandomForestOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0),

         # keyword arguments for building RandomForest oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(n_estimators=100,
                               max_depth=100,
                               max_features="auto"),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="hopper_controller/split",
                               is_absolute=False)))


register('HopperController-FullyConnected-v0',
         'design_bench.datasets.continuous.hopper_controller_dataset:HopperControllerDataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=512,
                               activation='relu',
                               num_layers=2,
                               epochs=20,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="hopper_controller/split",
                               is_absolute=False)))


register('Superconductor-GP-v0',
         'design_bench.datasets.continuous.superconductor_dataset:SuperconductorDataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=80,
             min_percentile=0),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(kernel=ConstantKernel(
                 constant_value=1.0, constant_value_bounds=(1e-9, 10.0)) *
                 RBF(length_scale=0.5, length_scale_bounds=(1e-9, 10.0)) +
                 RBF(length_scale=2.0, length_scale_bounds=(1e-9, 10.0))),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=2000,
                               to_disk=False,
                               disk_target=None,
                               is_absolute=None)))


register('Superconductor-RandomForest-v0',
         'design_bench.datasets.continuous.superconductor_dataset:SuperconductorDataset',
         'design_bench.oracles.sklearn:RandomForestOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=80,
             min_percentile=0),

         # keyword arguments for building RandomForest oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(n_estimators=100,
                               max_depth=100,
                               max_features="auto"),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="superconductor/split",
                               is_absolute=False)))


register('Superconductor-FullyConnected-v0',
         'design_bench.datasets.continuous.superconductor_dataset:SuperconductorDataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=80,
             min_percentile=0),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=512,
                               activation='relu',
                               num_layers=2,
                               epochs=5,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=5000,
                               to_disk=True,
                               disk_target="superconductor/split",
                               is_absolute=False)))


register('AntMorphology-Exact-v0',
         'design_bench.datasets.continuous.ant_morphology_dataset:AntMorphologyDataset',
         'design_bench.oracles.exact:AntMorphologyOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=40,
             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('AntMorphology-GP-v0',
         'design_bench.datasets.continuous.ant_morphology_dataset:AntMorphologyDataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=40,
             min_percentile=0),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(kernel=ConstantKernel(
                 constant_value=1.0, constant_value_bounds=(0.0, 10.0)) *
                 RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) +
                 RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=25000,
                               to_disk=False,
                               disk_target=None,
                               is_absolute=None)))


register('AntMorphology-RandomForest-v0',
         'design_bench.datasets.continuous.ant_morphology_dataset:AntMorphologyDataset',
         'design_bench.oracles.sklearn:RandomForestOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=40,
             min_percentile=0),

         # keyword arguments for building RandomForest oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(n_estimators=100,
                               max_depth=100,
                               max_features="auto"),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=25000,
                               to_disk=False,
                               disk_target=None,
                               is_absolute=None)))


register('AntMorphology-FullyConnected-v0',
         'design_bench.datasets.continuous.ant_morphology_dataset:AntMorphologyDataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=40,
             min_percentile=0),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=512,
                               activation='relu',
                               num_layers=2,
                               epochs=5,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=25000,
                               to_disk=False,
                               disk_target=None,
                               is_absolute=None)))


register('DKittyMorphology-Exact-v0',
         'design_bench.datasets.continuous.dkitty_morphology_dataset:DKittyMorphologyDataset',
         'design_bench.oracles.exact:DKittyMorphologyOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=40,
             min_percentile=0),

         # keyword arguments for building the exact oracle
         oracle_kwargs=dict(
             noise_std=0.0))


register('DKittyMorphology-GP-v0',
         'design_bench.datasets.continuous.dkitty_morphology_dataset:DKittyMorphologyDataset',
         'design_bench.oracles.sklearn:GaussianProcessOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=40,
             min_percentile=0),

         # keyword arguments for building GP oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(kernel=ConstantKernel(
                 constant_value=1.0, constant_value_bounds=(0.0, 10.0)) *
                 RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) +
                 RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=25000,
                               to_disk=False,
                               disk_target=None,
                               is_absolute=None)))


register('DKittyMorphology-RandomForest-v0',
         'design_bench.datasets.continuous.dkitty_morphology_dataset:DKittyMorphologyDataset',
         'design_bench.oracles.sklearn:RandomForestOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=40,
             min_percentile=0),

         # keyword arguments for building RandomForest oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=2000,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(n_estimators=100,
                               max_depth=100,
                               max_features="auto"),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.5,
                               subset=None,
                               shard_size=25000,
                               to_disk=False,
                               disk_target=None,
                               is_absolute=None)))


register('DKittyMorphology-FullyConnected-v0',
         'design_bench.datasets.continuous.dkitty_morphology_dataset:DKittyMorphologyDataset',
         'design_bench.oracles.tensorflow:FullyConnectedOracle',

         # keyword arguments for building the dataset
         dataset_kwargs=dict(
             max_samples=None,
             distribution=None,
             max_percentile=40,
             min_percentile=0),

         # keyword arguments for training FullyConnected oracle
         oracle_kwargs=dict(
             noise_std=0.0,
             max_samples=None,
             distribution=None,
             max_percentile=100,
             min_percentile=0,

             # parameters used for building the model
             model_kwargs=dict(hidden_size=512,
                               activation='relu',
                               num_layers=2,
                               epochs=5,
                               shuffle_buffer=5000,
                               learning_rate=0.001),

             # parameters used for building the validation set
             split_kwargs=dict(val_fraction=0.1,
                               subset=None,
                               shard_size=25000,
                               to_disk=False,
                               disk_target=None,
                               is_absolute=None)))
