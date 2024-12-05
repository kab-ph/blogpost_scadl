# SCADL: A new tool from the Donjon related to side-channel attacks (SCAs) using deep learning (DL)

In 2019, Ledger Donjon (Ledger's product security team) released [lascar](https://github.com/Ledger-Donjon/lascar) as a SCA tool. 
Since then, Deep Learning based SCAs (DL-SCAs) research have seen significant progress by several interesting publications. 
During our research activities, we developed a new tool implementing the most recent DL-SCAs methods to help us during side-channel evaluations.

We are pleased to release [scadl](https://github.com/Ledger-Donjon/scadl) which is a new in-house tool for performing SCAs using DL.
Following our goal to open source our work, we hope this project will help students, security researchers, and security experts in evaluation labs. Therefore, we expect your contribution.

 
## Introduction

DL-SCAs entered the field in recent years with the promise of more competitive performance compared to other techniques.
A lot of research papers proved the ability of such techniques to break protected cryptographic implementations with common side-channel countermeasures such as masking, jitter, and random delays insertion. 
In order to keep up with this research trend, we integrated the following techniques into [scadl](https://github.com/Ledger-Donjon/scadl). 

- Normal profiling: A straightforward profiling technique as the attacker will use a known-key dataset to train a DL model. 
Then, this model is used to attack the unknown-key data set. This technique was presented by the following work: [1](https://eprint.iacr.org/2016/921) and [2](https://eprint.iacr.org/2018/053). 
The authors showed also the strength of such technique against protected designs with jitter and masking.

- [Non-profiling](https://tches.iacr.org/index.php/TCHES/article/view/7387): In order to take the advantage of DL attacks against protected designs,  this technique is similar to differential power analysis ([DPA](https://paulkocher.com/doc/DifferentialPowerAnalysis.pdf))
but it has several advantages over DPA to attack protected designs (masking and desynchronization) because of using the accuracy of the DL model instead of any statistical-based method such as DPA that requires traces processing.

- [Multi-label](https://eprint.iacr.org/2020/436): A technique to attack multiple keys using only one DL model.  

- [Multi-tasking](https://eprint.iacr.org/2023/006.pdf): Another technique for attacking multiple keys using a single model.

- Data augmentation: A technique to increase the dataset to boost the DL efficiency. Scadl includes [mixup](https://eprint.iacr.org/2021/328.pdf) and [random-crop](https://blog.roboflow.com/why-and-how-to-implement-random-crop-data-augmentation/).

- [Attribution methods](https://eprint.iacr.org/2019/143.pdf): This technique is used to reverse the DL model to understand how the model behaves during the prediction phase. It helps in improving the performance of the DL model. Moreover, it can be used as a leakage detection technique.

## Tutorials
The repository provides several [tutorials](https://github.com/Ledger-Donjon/scadl/tree/master/tutorial) as examples to use for each technique.


## Datasets
Scadl uses two different datasets for its tutorial. The first dataset is collected by running a non-protected AES on [ChipWhisperer-Lite](https://rtfm.newae.com/Targets/CW303%20Arm/). The second dataset is [ASCAD](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1) which is widely used in the side-channel attacks (SCAs) domain.

| ![ChipWhisperer power consumption trace](images/cw_aes_single.png)|
|:--:|
| *Power consumption traces acquired from a ChipWhisperer* |

## Example

As we mentioned before, scadl implements different types of DL-based attacks and here is an example of how to use scadl for non-profiling DL in case of ASCAD dataset.
The following example shows how non-profiling DL-SCAs can be performed using scadl in case of using ASCAD dataset.

- As a first step, we construct a DL model based on MLP. This model contains 2 layers in addition to the last layer.

```python
def mlp_ascad(len_samples: int) -> keras.Model:
    """It returns an mlp model"""
    model = Sequential()
    model.add(Input(shape=(len_samples,)))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    return model

```

- Then,  we construct a leakage model. It's a masked design and we deliver the time samples of manipulating the mask with the sbox output.
Therefore, we use the following leakage model, which is based on the least significant bit.

```python
def leakage_model(data: np.ndarray, guess: int) -> int:
    """It calculates lsb"""
    return 1 & ((sbox[data["plaintext"][TARGET_BYTE] ^ guess]))
```

- After that, we perform pre-processing on the leakage traces by normalization and subtracting the average to reduce the complexity of the DL model.

```python
x_train = normalization(remove_avg(leakages), feature_range=(-1, 1))
```

- The final step includes brute-forcing the unknown key and calculating the model accuracy/loss for each guessed key. 
The correct key should give the highest accuracy (or the lowest loss). This process is performed under a certain number of epochs which can be varied depending on the efficiency of the used DL model.

```python
EPOCHS = 10
guess_range = range(0, 256)
acc = np.zeros((len(guess_range), EPOCHS))
profile_engine = NonProfile(leakage_model=leakage_model)
for index, guess in enumerate(tqdm(guess_range)):
    acc[index] = profile_engine.train(
        model=mlp_ascad(x_train.shape[1]),
        x_train=x_train,
        metadata=metadata,
        hist_acc="accuracy",
        guess=guess,
        num_classes=2,
        epochs=EPOCHS,
        batch_size=1000,
        verbose=0
    )
guessed_key = np.argmax(np.max(acc, axis=1))
print(f"guessed key = {guessed_key}")

```

- The following figure shows the accuracy of all the brute-forced keys. The black curve indicates the accuracy of the correct guessed key.

![cw_trace](images/non_profiling_result.png)

## Takeaways

We present an in-house open-source tool for performing SCAs using DL, in order to help security professionals to understand and perform the most recent
techniques. 

## References

1. B. Timon, [Non-Profiled Deep Learning-based Side-Channel attacks with Sensitivity Analysis
](https://tches.iacr.org/index.php/TCHES/article/view/7387), CHES, 2019.
2. H. Maghrebi, [Deep Learning based Side-Channel Attack: a New Profiling Methodology based on Multi-Label Classification
](https://eprint.iacr.org/2020/436), Cryptology ePrint Archive, 2020.
3. B. Hettwer et al., [Deep Neural Network Attribution Methods for
Leakage Analysis and Symmetric Key Recovery](https://eprint.iacr.org/2019/143), Cryptology ePrint Archive, 2019.
4. T. Marquet et al., [Exploring Multi-Task Learning in the Context of
Masked AES Implementations](https://eprint.iacr.org/2023/006), COSADE, 2024.
5. K. Abdellatif, [Mixup Data Augmentation for Deep Learning
Side-Channel Attacks](https://eprint.iacr.org/2021/328), Cryptology ePrint Archive, 2021.

**Done by Karim M. Abdellatif and Leo Benito**

















