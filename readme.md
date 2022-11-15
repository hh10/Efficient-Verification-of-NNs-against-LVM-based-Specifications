# Efficient Verification of NNs against LVM-based Specifications

This is the codebase for the verification pipeline produced and experiments in our submission. 
## Installation:
To be able to use the codebase, install gurobi (make sure its executable and libraries can be found, can run its setup.py) and all packages in requirements.txt. Run `python3 -m tests.tests` to ensure that tests run.
All verification based results in Table 1 and 4 requires VeriNet backend, which can be downloaded from its author's [github repo](https://github.com/vas-group-imperial/VeriNet). After its setup, make sure that `python verinet_line_segment_verification.py` runs successfully.
## Usage:
Commands for various uses of this codebase are as follows:
1. To train the verification pipeline elements:
`python3 train.py --config_file=*<path to config file>*`
    * Config file should have a dict with the following parameters in a dict (see [sample_configs/*](sample_configs) for reference):
        >`# Dataset related params` \
            "dataset": *MNIST/FashionMNIST/CelebA/TrafficSignsDynSynth/Objects3d* \
            "input_shape": *[C, H, W]* \
            "data_balance_method": *None/reduce/increase* \
            "classes": *'default' for standard torch datasets, list of attribute names for CelebA, 'dir_based' for custom datasets* \
            "conditional": *list/dict specifying the classes/attributes to condition on* \
            "batch_size": *xx* \
        `# Encoding head related params` \
            "latent_dim": *xx* \
            "loss_kl_weight": *xx* \
            "loss_conditional_weight": *xx* \
            "loss_recons_weight": *xx* \
            "conditional_loss_func": *[CE/MSE/Generator]* \
        `# Learning related params`\
            "num_epochs": *xx* \
            "lr": *xx* \
            "train_cla": *when verifying pretrained classifiers, whether to also train them or not* \
            "only_cla_training_epochs": *last x epochs to only train the classifier* \
            "only_vae_training_epochs": *\# of epochs to only train the vae (relevant when training both classifier and VAE)* \
            "GAN_start_training_epochs": *Epochs from when to start also optimising against GAN loss* \
        `# Classifier to be verified` \
            "model": *specify params to construct the subnetworks like classifier, VAE, etc.* \
        `# Run identifiers` \
            "desc": *short description added to summary tag and results folder* \
            "notes": *longer detailed notes*

2. To verify a trained classifier (pipeline) for different datasets:
    * MNIST, FashionMNIST, TrafficSignsDynSynth:
        * For every image i in dataset, verifies for an input set [i, transform(i)] for all transform in test_transforms (till the range and in steps specified): \
            `python3 verify.py --model_path=<path to resp. model.tar>`
    * CelebA:
        * For an image in dataset, verifies invariance against a test_attribute or against head tilt: \
            `python3 verify.py --model_path=<path to CelebA_model.tar> --test_attribute=<any attribute> --flip_head=True/False`
    * Object10:
        * Given transforms {tA, tB}, for an image i, verifies for an input set [tA(i), tB(i)]: \
            `python3 verify.py --model_path=<path to Object10_model.tar> --test_attribute=<attrA> --target_attributes=<attrB,attrC>` \
    For each dataset image, the script also verifies for eps in conditional dimensions if any.

3. Reproducing experiment results in the submission:
    - All reported networks are defined in [model.py](model.py), [models_impl/](models_impl) and notebook/notebook_utils.py.
    - Most reconstruction outcomes in Table 1 can be reproduced from a straightforward run of the EDC_SRVP_pipelines_and_decoders_comparison.ipynb notebook. Some reported SRVP pipelines requires training as explained in 1. using configs in [sample_configs](sample_configs). Some trained pipelines from this run are uploaded as release with the repo, for user to run the verification scripts in the next step directly.
    - (Requires VeriNet installation) All verification outcomes in Table 1 and in Appendix can be reproduced by running [bounds_computation.py](bounds_computation.py) and notebook/verification_comparison.py on the trained pipelines. Some verification results require a verification run as explained in 2. above. 
