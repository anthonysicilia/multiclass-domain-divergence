# multiclass-domain-divergence
This is a shared code repository for the papers:
[The Change that Matters in Discourse Parsing: Estimating the Impact of Domain Shift on Parser Error](https://arxiv.org/abs/2203.11317) to appear 
in Findings of [ACL 2022](https://www.2022.aclweb.org) and [PAC-Bayesian Domain Adaptation Bounds for Multiclass Learners](https://openreview.net/pdf?id=S0lx6I8j9xq) to appear in 
[UAI 2022](https://www.auai.org/uai2022/). A package derived from this code is available [here](https://github.com/anthonysicilia/classifier-divergence).

Please, consider citing these papers if you use this code.

## Relevant Links
arXiv (ACL 2022): https://arxiv.org/abs/2203.11317

OpenReview (UAI 2022): https://openreview.net/pdf?id=S0lx6I8j9xq

shared code: https://github.com/anthonysicilia/multiclass-domain-divergence

UAI code: https://github.com/anthonysicilia/pacbayes-adaptation-UAI2022

ACL code: https://github.com/anthonysicilia/change-that-matters-ACL2022

## Running this code
For ease of use, we have created a python script ```experiments/make_scripts.py``` to generate example bash scripts. In many cases, these may be exactly identical to the scripts used to generate results in the paper. Albeit, we encourage second checking the accompanying manuscripts to verify parameters etc. The bash scripts interfrace with the experiments module to create the raw results for each experiment. Following this, code for summarizing raw results can be run using ```experiments/results.py```. The latter script has undergone a number of different iterations (i.e., for each individual paper). The comments on this script may help to identify relevant code. As a final note, ```experiments``` is a module, so scripts should be run accordingly. Feel free to contact us with any questions (e.g., by raising an issue here or using the contact information available in the accompanying papers).

