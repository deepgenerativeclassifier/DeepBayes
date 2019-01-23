# DeepBayes
We provide an example code for readers to explore attacks on generative classifiers. 

Depending on the configuration of your machine, and/or possible bad local optima issues, results might be slightly different from those reported in the paper.

We expect the general conclusions to stay the same.

You need to have the following packages installed:
Tensorflow, Keras

On our machine the versions are: Tensorflow 1.10.1, Keras 2.2.2

Please do **NOT** use the latest cleverhans package. Use the cleverhans package included in this repo instead.
This is because we are sampling multiple latent variables in prediction time, which requires specific tensor shape matching in the code.


## Usage:

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


Once finished, to test an attack, run

    cd test_attacks/
    python attack.py --data mnist --attack fgsm --eps 0.1 --victim bayes_K10_A --save

Here, the configs are:

--data: mnist

--attack: the type of attack to run, should be one of {fgsm, pgd, mim, cw}.

--eps: the distortion strength for L_inf attacks, or the c parameter for CW, should be a positive value.

--victim: the classifier in test, here "bayes_K10_A" means we will run the Bayes rule based classification method with K=10 samples, and the classifier type is "A".

If you add in --save, crafted adversarial examples will be saved. No need for this config if you don't want to save the attacks.

During the attack, the program will print something for progress, and finally print the sucess rate for this attack.
Also you can change the attack settings by editing values in attack_config.py


When the attack is finished and saved, you can test the detection by running

    python detect_attacks_logp.py --victim bayes_K10_A_cnn --guard bayes_K10_A --attack fgsm_eps0.10 --data mnist --save

Here the configs are:

--victim: the victim classifier that the adversarial examples are crafted on. Should include suffix "_cnn" here.

--guard: the classifier that is used to test the adversarial examples. When "victim" and "guard" differ, this will test transferred attacks.

--attack: the attack and the parameters of that attack. For {fgsm, pgd, mim} it should look like "[attack]_eps[%2.f]", for cw it should look like "cw_c[%.2f]_lr[%.3f]"

--data: mnist

if you add in --save, detection results will be saved. No need for this config if you don't want to save the detection results.

During detection, the program will print classification accuracy of the guard classifier, statistics of the attack and the data, and detection rates.

If not testing transferred attacks, for classification accuracy, it will print "(all/victim) = [accuracy%] / 100%", and the "100%" here does not have a specific meaning.

For transferred attacks, "(all/victim) = [accuracy%] / [accuracy%]" prints the accuracy on **all** adv inputs and the accuracy on **successful attacks on the victim**.

For discriminative classifiers, the program still computes the logit/marginal detection rates (because it can).
However we don't expect the logit/marginal detection to work well on discriminative classifiers, since the semantic meaning of these logits are unclear.
