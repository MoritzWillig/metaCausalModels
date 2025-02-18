# Systems with Switching Causal Relations: A Meta-Causal Perspective

Code Repository for the corresponding paper "*Systems with Switching Causal Relations: A Meta-Causal Perspective*" accepted at ICLR 2025.

**Authors**: Moritz Willig, Tim Nelson Tobiasch, Florian Peter Busch, Jonas Seng, Devendra Singh Dhami, Kristian Kersting

**OpenReview**: [https://openreview.net/forum?id=J9VogDTa1W](https://openreview.net/forum?id=J9VogDTa1W)

## Abstract

Most works on causality in machine learning assume that causal relationships are governed by a constant underlying process. However, the flexibility of agents' actions or tipping point behavior in the environmental process can change the qualitative dynamics of the system. As a result, new causal relationships may emerge, while existing ones change or disappear, resulting in an altered causal graph. To analyze these qualitative changes on the causal graph, we propose the concept of meta-causal states, which groups classical causal models into clusters based on equivalent qualitative behavior and consolidates specific mechanism parameterizations. We demonstrate how meta-causal states can be inferred from observed agent behavior, and discuss potential methods for disentangling these states from unlabeled data. Finally, we direct our analysis toward the application of a dynamical system, demonstrating that meta-causal states can also emerge from inherent system dynamics, and thus constitute more than a context-dependent framework in which mechanisms emerge merely as a result of external factors.


## Requirements
All experiments where conducted using Python 3.8 .  
Install the necessary requirements `pip install -r requirements.txt`

## Experiments

Scripts are located in the `metaCausal` folder.
* `A1_1_test_algo_convergence.py` - evaluates the convergence rates for single runs of the EM algorithm.
* `A2_1_test_algo_decomposition.py` - evaluates the confusion matrix for determining the number of mechanisms between variables in the bivariate case.
* `V1_stability_region.py` - Plots the stability region (Fig. 2, center)
* `V2_sigm.py` - Plots the Sigmoidal function (Fig. 2, left)
* `V3_plot_samples.py` - Generates data sample plots (Fig. 3)


## Cite

```
@inproceedings{willig2025systems,
title={Systems with Switching Causal Relations: A Meta-Causal Perspective},
author={Moritz Willig and Tim Tobiasch and Florian Peter Busch and Jonas Seng and Devendra Singh Dhami and Kristian Kersting},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=J9VogDTa1W}
}
```


## License

This repository is licensed under MIT License. See [./LICENSE](./LICENSE) for full license text.
