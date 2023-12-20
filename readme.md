# Multiple Hypothesis Dropout

Official implementation of the paper: 
 [**Multiple Hypothesis Dropout: Estimating the Parameters of Multi-Modal Output Distributions**](https://arxiv.org/abs/2312.11735)<br>
 [David D. Nguyen](https://www.linkedin.com/in/dngu/), [David Liebowitz](https://www.linkedin.com/in/david-liebowitz/), [Surya Nepal](https://people.csiro.au/N/S/Surya-Nepal), [Salil Kanhere](https://www.unsw.edu.au/staff/salil-kanhere)



<hr />

## News
* **10 December, 2023**: Paper accepted into AAAI 2024 Main Track.

## Paper Contrbutions

- We introduce the **Multiple Hypothesis Dropout** (MH-Dropout), a novel variant of dropout that converts a single-output function into a multi-output function using the subnetworks derived from a base neural network.
-  We found that combining Winner-Takes-All loss with *stochastic hypothesis sampling* allows MH-Dropout networks to stably learn the statistical variability of targets in multi-output scenarios.
-  We describe a **Mixture of Multiple-Output Functions** (MoM), composed of MH-Dropout networks to address multi-modal output distributions in supervised learning settings. We show this architecture can learn the parameters of the components of a Gaussian mixture. 
-  We propose a novel MH-VQVAE that employs MH-Dropout networks to estimate the variance of clusters in embedding representational space. We show this approach significantly improves codebook efficiency and generation quality.

## Codebase Description
- Implementations of Multiple Hypothesis Dropout and Mixture of Multiple-Output Functions (MoM) can be found in the model folder.
- The notebooks reproduce experiments presented in the paper.

## Requirements
- Please install pytorch==1.12 and the packages listed under requirements.txt.

## Citation
If you found our work useful, please consider citing us:

```bibtex
@misc{nguyen2023multiple,
      title={Multiple Hypothesis Dropout: Estimating the Parameters of Multi-Modal Output Distributions}, 
      author={David D. Nguyen and David Liebowitz and Surya Nepal and Salil S. Kanhere},
      year={2023},
      eprint={2312.11735},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contact
For inquiries or issues, kindly reach out by creating an issue in the repository or by contacting david.nguyen (at) data61.csiro.au or david.nguyen (at) proton.me.


## License
Copyright © Cyber Security Research Centre Limited 2023. This work has been supported by the Cyber Security Research Centre (CSCRC) Limited whose activities are partially funded by the Australian Government’s Cooperative Research Centres Programme. We are currently tracking the impact CSCRC funded research. If you have used this code/data in your project, please contact us at contact@cybersecuritycrc.org.au to let us know.






