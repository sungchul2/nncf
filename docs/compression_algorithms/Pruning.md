# Pruning


## Filter pruning

Filter pruning algorithm zeros output filters in Convolutional layers based on some filter importance criterion  (filters with smaller importance are pruned).
The framework contains three filter importance criteria: `L1`, `L2` norm, and `Geometric Median`. Also, different schemes of pruning application are presented by different schedulers.
Not all Convolution layers in the model can be pruned. Such layers are determined by the model architecture automatically as well as cross-layer dependencies that impose constraints on pruning filters.

### Filter importance criteria **L1, L2**

 `L1`, `L2` filter importance criteria are based on the following assumption:
> Convolutional filters with small $l_p$ norms do not significantly contribute to output activation values, and thus have a small impact on the final predictions of CNN models.
In the above, the $l_p$ norm for filter $F$ is:

$$||F||_p = \sqrt[p]{\sum_{c, k_1, k_2 = 1}^{C, K, K}|F(c, k_1, k_2)|^p}$$

During the pruning procedure filters with smaller  `L1` or `L2` norm will be pruned first.

### Geometric Median

Usage of the geometric median filter importance criterion is based on the following assumptions:
> Let $\{F_i, \dots , F_j\}$ be the set of $N$ filters in a convolutional layer that are closest to the geometric median of all the filters in that layer.   As it was shown, each of those filters can be decomposed into a linear combination of the rest of the filters further from the geometric median with a small error. Hence, these filters can be pruned without much impact on network accuracy.  Since we have only fixed number of filters in each layer and the task of calculation of geometric median is a non-trivial problem in computational geometry, we can instead find which filters minimize the summation of the distance with other filters.

Then Geometric Median importance of $F_i$ filter from $L_j$ layer is:

$$G(F_i) = \sum_{F_j \in \{F_1, \dots F_m\}, j\neq i} ||F_i - F_j||_2$$

where $L_j$ is $j$-th convolutional layer in model and $\{F_1, \dots F_m\} \in L_j$ is a set of all  output filters in $L_j$ layer.  
Then during pruning filters with smaller $G(F_i)$ importance function will be pruned first.

---
## Schedulers

### Baseline Scheduler

1. During `num_init_steps` epochs, the model is trained without pruning.
2. The pruning algorithm calculates filter importances and prunes a `pruning_target` part of the filters with the smallest importance in each prunable convolution. The zeroed filters are frozen afterwards and the remaining model parameters are fine-tuned.

**Parameters of the scheduler** :
- `num_init_steps` - number of epochs for model pretraining **before** pruning.
- `pruning_target` - pruning rate target. For example, the value `0.5` means that right after pretraining, convolutions that can be pruned will have 50% of their filters set to zero.

### Exponential scheduler

1. Similar to the Baseline scheduler, during `num_init_steps` epochs model is pretrained without pruning.
2. During the next `pruning steps` epochs, `Exponential scheduler` gradually increasing pruning rate from `pruning_init` to `pruning_target`. 
    - After each pruning training epoch, pruning algorithm calculates filter importances for all convolutional filters and prune (setting to zero) `current_pruning_rate` part of filters with the smallest importance in each Convolution.
4. After `num_init_steps` + `pruning_steps` epochs, algorithm with zeroed filters is frozen and remaining model parameters only fine-tunes.

Current pruning rate $P_{i}$ (on $i$-th epoch) during training calculates by equation:

$$P_i = a * e^{- k * i}$$

where $a$ and $k$ are parameters.

**Parameters of scheduler** :
- `num_init_steps` - number of epochs for model pretraining before pruning.
- `pruning_steps` - the number of epochs during which the pruning rate target is increased from `pruning_init` to `pruning_target` value.
- `pruning_init` - initial pruning rate target. For example, value `0.1` means that at the begging of training, convolutions that can be pruned will have 10% of their filters set to zero.
- `pruning_target` - pruning rate target at the end of the schedule. For example, the value `0.5` means that at the epoch with the number of `num_init_steps + pruning_steps`, convolutions that can be pruned will have 50% of their filters set to zero.

### Exponential with bias scheduler
Similar to the `Exponential scheduler`, but current pruning rate $P_{i}$ (on $i$-th epoch) during training calculates by equation:

$P_i = a * e^{- k * i} + b$$

where $a$, $k$, and $b$ are parameters.

> **NOTE** :  Baseline scheduler prunes filters only ONCE and after it just fine-tunes remaining parameters while exponential (and exponential with bias) schedulers choose and prune different filters subsets at each pruning epoch.

---
## Batch-norm statistics adaptation

After the compression-related changes in the model have been committed, the statistics of the batchnorm layers (per-channel rolling means and variances of activation tensors) can be updated by passing several batches of data through the model before the fine-tuning starts. 
This allows to correct the compression-induced bias in the model and reduce the corresponding accuracy drop even before model training. 
This option is common for quantization, magnitude sparsity and filter pruning algorithms. 
It can be enabled by setting a non-zero value of `num_bn_adaptation_samples` in the `batchnorm_adaptation` section of the `initializer` configuration (see example below).


## Interlayer ranking types

Interlayer ranking type can be one of `unweighted_ranking` or `learned_ranking`.
- In case of `unweighted_ranking` and with `all_weights=True`, all filter norms will be collected together and sorted to choose the least important ones. But this approach may not be optimal because filter norms are a good measure of filter importance inside a layer, but not across layers.
- In the case of `learned_ranking` that uses re-implementation of [Learned Global Ranking method](https://arxiv.org/abs/1904.12368), a set of ranking coefficients will be learned for comparing filters across different layers.
The $(a_i, b_i)$ pair of scalars will be learned for each $i$-th layer and used to transform norms of $i$-th layer filters before sorting all filter norms together as $a_i * N_i + b_i$, where $N_i$ is vector of filter norms of $i$-th layer, $(a_i, b_i)$ is ranking coefficients for $i$-th layer.
This approach allows pruning the model taking into account layer-specific sensitivity to weight perturbations and get pruned models with higher accuracy.

**Filter pruning configuration file parameters** :
```
{
    "algorithm": "filter_pruning",
    "initializer": {
        "batchnorm_adaptation": {
            "num_bn_adaptation_samples": 2048, // Number of samples from the training dataset to pass through the model at initialization in order to update batchnorm statistics of the original model. The actual number of samples will be a closest multiple of the batch size.
        }
    },
    "pruning_init": 0.1, // Initial value of the pruning level applied to the convolutions that can be pruned in 'create_compressed_model' function. 0.0 by default.
    "params": {
        "schedule": "exponential", // The type of scheduling to use for adjusting the target pruning level. Either `exponential`, `exponential_with_bias`,  or `baseline`, by default it is `exponential`"
        "pruning_target": 0.4, // Target value of the pruning level for the convolutions that can be pruned. These convolutions are determined by the model architecture. 0.5 by default.
        "pruning_flops_target": 0.4, // Target value of the pruning level by FLOPs in the whole model. Only one parameter from `pruning_target` and `pruning_flops_target` can be set. If none of them is specified, `pruning_target` = 0.5 is used as the default value. 
        "num_init_steps": 3, // Number of epochs for model pretraining before starting filter pruning. 0 by default.
        "pruning_steps": 10, // Number of epochs during which the pruning rate is increased from `pruning_init` to `pruning_target` value.
        "filter_importance": "L2", // The type of filter importance metric. Can be one of `L1`, `L2`, `geometric_median`. `L2` by default.
        "interlayer_ranking_type": "unweighted_ranking", // The type of filter ranking across the layers. Can be one of `unweighted_ranking`, `learned_ranking`. `unweighted_ranking` by default.
        "all_weights": false, // Whether to prune layers independently (choose filters with the smallest importance in each layer separately) or not. `False` by default.
        "prune_first_conv": false, // Whether to prune first Convolutional layers or not. First means that it is a convolutional layer such that there is a path from model input to this layer such that there are no other convolution operations on it. `False` by default (`True` by default in case of 'learned_ranking' interlayer_ranking_type).
        "prune_downsample_convs": false, // Whether to prune downsample Convolutional layers (with stride > 1) or not. `False` by default (`True` by default in case of 'learned_ranking' interlayer_ranking_type).
        "prune_batch_norms": true, // Whether to nullifies parameters of Batch Norm layer corresponds to zeroed filters of convolution corresponding to this Batch Norm. `True` by default.
        "save_ranking_coeffs_path": "path/coeffs.json", // Path to save .json file with interlayer ranking coefficients.
        "load_ranking_coeffs_path": "PATH/learned_coeffs.json", // Path to loading interlayer ranking coefficients .json file, pretrained earlier.
        "legr_params": { // Set of parameters, that can be set for 'learned_ranking' interlayer_ranking_type case
            "generations": 200, //  Number of generations for evolution algorithm optimizing. 400 by default
            "train_steps": 150, // Number of training steps to estimate pruned model accuracy. 200 by default 
            "max_pruning": 0.6, // Pruning level for the model to train LeGR algorithm on it. If learned ranking will be used for multiple pruning rates, the highest should be used as `max_pruning`. If model will be pruned with one pruning rate, target pruning rate should be used.
            "random_seed": 42, // Random seed for ranking coefficients generation during optimization 
        },
    },

    // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'denylist'. Optional.
    "ignored_scopes": []

    // A list of model control flow graph node scopes to be considered for this operation - functions as a 'allowlist'. Optional.
    // "target_scopes": []
}
```

> **NOTE** : In all our pruning experiments we used SGD optimizer.
