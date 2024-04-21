from morphing_agents.mujoco.dkitty.env import MorphingDKittyEnv
from morphing_agents.mujoco.dkitty.elements import LEG
from morphing_agents.mujoco.dkitty.elements import LEG_LOWER_BOUND
from morphing_agents.mujoco.dkitty.elements import LEG_UPPER_BOUND
from design_bench.oracles.exact_oracle import ExactOracle
from design_bench.datasets.continuous_dataset import ContinuousDataset
from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
from design_bench.disk_resource import DiskResource, SERVER_URL
import numpy as np
import pickle as pkl


class DKittyMorphologyOracle(ExactOracle):
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

    """

    name = "exact_average_return"

    @classmethod
    def supported_datasets(cls):
        """An attribute the defines the set of dataset classes which this
        oracle can be applied to forming a valid ground truth score
        function for a model-based optimization problem

        """

        return {DKittyMorphologyDataset}

    @classmethod
    def fully_characterized(cls):
        """An attribute the defines whether all possible inputs to the
        model-based optimization problem have been evaluated and
        are are returned via lookup in self.predict

        """

        return False

    @classmethod
    def is_simulated(cls):
        """An attribute the defines whether the values returned by the oracle
         were obtained by running a computer simulation rather than
         performing physical experiments with real data

        """

        return True

    def protected_predict(self, x, render=False, **render_kwargs):
        """Score function to be implemented by oracle subclasses, where x is
        either a batch of designs if self.is_batched is True or is a
        single design when self._is_batched is False

        Arguments:

        x_batch: np.ndarray
            a batch or single design 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned

        Returns:

        y_batch: np.ndarray
            a batch or single prediction 'y' made by the oracle model,
            corresponding to the ground truth score for each design
            value 'x' in a model-based optimization problem

        """

        # create a policy forward pass in numpy
        def mlp_policy(h):
            h = np.maximum(0.0, self.policy[
                "base_network.network.0.weight"] @ h + self.policy[
                "base_network.network.0.bias"])
            h = np.maximum(0.0, self.policy[
                "base_network.network.2.weight"] @ h + self.policy[
                "base_network.network.2.bias"])
            return np.tanh(np.split(self.policy[
                "base_network.network.4.weight"] @ h + self.policy[
                "base_network.network.4.bias"], 2)[0])

        # convert vectors to morphologies
        env = MorphingDKittyEnv(expose_design=True,
                                normalize_design=True,
                                fixed_design=[
            LEG(*np.clip(np.array(xi), LEG_LOWER_BOUND,
                         LEG_UPPER_BOUND)) for xi in np.split(x, 4)])

        # do many rollouts using a pretrained agent
        obs = env.reset()
        sum_of_rewards = np.zeros([1], dtype=np.float32)
        for t in range(self.rollout_horizon):
            obs, rew, done, info = env.step(mlp_policy(obs))
            if render:
                env.render(**render_kwargs)
            sum_of_rewards += rew.astype(np.float32)
            if done:
                break

        # we average here so as to reduce randomness
        return sum_of_rewards

    def __init__(self, dataset: ContinuousDataset,
                 rollout_horizon=100, **kwargs):
        """Initialize the ground truth score function f(x) for a model-based
        optimization problem, which involves loading the parameters of an
        oracle model and estimating its computational cost

        Arguments:

        dataset: DiscreteDataset
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        noise_std: float
            the standard deviation of gaussian noise added to the prediction
            values 'y' coming out of the ground truth score function f(x)
            in order to make the optimization problem difficult
        internal_measurements: int
            an integer representing the number of independent measurements of
            the prediction made by the oracle, which are subsequently
            averaged, and is useful when the oracle is stochastic

        """

        # the number of transitions per trajectory to sample
        self.rollout_horizon = rollout_horizon

        # ensure the trained policy has been downloaded
        policy = "dkitty_morphology/dkitty_oracle.pkl"
        policy = DiskResource(
            policy, is_absolute=False, download_method="direct",
            download_target=policy)
        if not policy.is_downloaded and not policy.download():
            raise ValueError("unable to download trained policy for ant")

        # load the weights of the policy
        with open(policy.disk_target, "rb") as f:
            self.policy = pkl.load(f)

        # initialize the oracle using the super class
        super(DKittyMorphologyOracle, self).__init__(
            dataset, internal_batch_size=1, is_batched=False,
            expect_normalized_y=False,
            expect_normalized_x=False, expect_logits=None, **kwargs)
