# DeepBayes

We provide an example code for readers to explore attacks on generative classifiers. 

**Depending on the configuration of your machine, and/or possible bad local optima issues, results might be slightly different from those reported in the paper.**

We expect the general conclusions to stay the same.

You need to have the following packages installed:
Tensorflow, Keras

On our machine the versions are: Tensorflow 1.10.1, Keras 2.2.2

Please do **NOT** use the latest cleverhans package. Use the cleverhans package included in this repo instead.
This is because we are sampling multiple latent variables in prediction time, which requires specific tensor shape matching in the code.


## Train a generative classifier:

First run 

    python vae_mnist.py A

This will train classifier "A" on MNIST with specific settings detailed in vae_mnist.py. 
You can also try other classifiers (from A to G).

The corresponding graphical models of the classifiers are:

A: GFZ

B: GFY

C: DFZ

D: DFX

E: DBX

F: GBZ

G: GBY

If you want to train generative classifiers on the CIFAR-binary classification task, you need to first download the CIFAR-10 dataset from [this link](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), unzip it, then put the extracted folder into [cifar_data/](cifar_data/). Then simply run

    python vae_cifar_binary.py A

This will train a binary classifier based on the raw image inputs.

For the fusion model, you need to specify the type of the classifier and the layer of feature from VGG you want to use. First make sure you have downloaded the VGG weights for CIFAR-10 from [this repo](https://github.com/geifmany/cifar-vgg) and put the h5 file to [test_attack/load/vgg_model/](test_attack/load/vgg_model/). After that, run, for example,

    python vae_cifar_fea.py E mid

This will train a DBX classifier using mid-level VGG features (conv9). You can play around with classfier type **E, F, G** and feature level **low, mid, high**.


## Perform attacks on generative classifiers

Once a classifier is trained, to test an attack, run e.g.

    cd test_attacks/
    python attack.py --data mnist --attack fgsm --eps 0.1 --victim bayes_K10_A --save

Here, the configs are:

**--data**: should be one of {mnist, plane_frog, cifar10}, plane_frog refers to the CIFAR-binary dataset.

**--attack**: the type of attack to run, should be one of {fgsm, pgd, mim, cw}.

**--eps**: the distortion strength for L_inf attacks, or the c parameter for CW, should be a positive value.

**--victim**: the classifier in test, here "bayes_K10_A" means we will run the Bayes rule based classification method with K=10 samples, and the classifier type is "A".

If you add in **--save**, crafted adversarial examples will be saved. No need for this config if you don't want to save the attacks.

During the attack, the program will print something for progress, and finally print the success rate for this attack.
Also you can change the attack settings by editing values in attack_config.py


When the attack is finished and saved, you can test the detection by running

    python detect_attacks_logp.py --victim bayes_K10_A_cnn --guard bayes_K10_A --attack fgsm_eps0.10 --data mnist --save

Here the configs are:

**--victim**: the victim classifier that the adversarial examples are crafted on. Should include suffix "_cnn" here.

**--guard**: the classifier that is used to test the adversarial examples. When "victim" and "guard" differ, this will test transferred attacks.

**--attack**: the attack and the parameters of that attack. For {fgsm, pgd, mim} it should look like "[attack]_eps[%2.f]", for cw it should look like "cw_c[%.2f]_lr[%.3f]"

**--data**: should be one of {mnist, plane_frog, cifar10}, plane_frog refers to the CIFAR-binary dataset.

if you add in **--save**, detection results will be saved. No need for this config if you don't want to save the detection results.

During detection, the program will print classification accuracy of the guard classifier, statistics of the attack and the data, and detection rates.

If not testing transferred attacks, for classification accuracy, it will print "(all/victim) = [accuracy%] / 100%", and the "100%" here does not have a specific meaning.

For transferred attacks, "(all/victim) = [accuracy%] / [accuracy%]" prints the accuracy on **all** adv inputs and the accuracy on **successful attacks on the victim**.

For discriminative classifiers, the program still computes the logit/marginal detection rates (because it can).
However we don't expect the logit/marginal detection to work well on discriminative classifiers, since the semantic meaning of these logits are unclear.

## The WB+S+(M/L/K) attacks: "superwhite"

We also include the implementation of the white-box attack targeting randomness and detection. See [test_attack/superwhite.py](https://github.com/deepgenerativeclassifier/DeepBayes/blob/master/test_attacks/superwhite.py) for details.

To test this attack, after training your model, simply run e.g.

    python attack_superwhite.py --victim bayes_K10_F --eps 0.2 --data mnist --lbd 0.1 --save --snapshot

This will perform WB+S+L attack (against randomness + logit detection). To perform attacks agains marginal or KL detection, change the detection loss [here](https://github.com/deepgenerativeclassifier/DeepBayes/blob/master/test_attacks/superwhite.py#L213) in [test_attack/superwhite.py](https://github.com/deepgenerativeclassifier/DeepBayes/blob/master/test_attacks/superwhite.py).

Here, the extra configs are (others are the same as for attack.py):

**--lbd**: the lambda detect parameter that controls the weight of the detection loss.

**--snapshot**: if you add in this flag then the attack will consider randomness in the sampling procedure. Without this flag the attack is agnostic to the MC approximation for generative classifiers, similar to attack.py.

Then after attack, to compute the detection rates, run detect_attacks_log.py with flag --attack superwhite_eps0.20_lambda0.1

# Note on CIFAR-binary robustness results

For CIFAR-binary tasks we only report results **on clean test images that all the candidate classifiers have predicted the right labels**. This means **you need to train all the classifiers before running detect_attacks_log.py**. Also after the training of each classifier, you need to run, e.g.

    cd test_attack/
    python extract_index.py bayes_K10_A plane_frog

in order to extract and save the indices of the correctly classified test images. 

