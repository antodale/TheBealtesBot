# The BeatlesBot: A Song-Writing Aid Tool

**Author:** Antonio D'Alessandro (University of Bologna)
**Contact:** antonio.dalessandr10@studio.unibo.it
**Date:** June 2, 2025

## Overview
The BeatlesBot is a functional tool designed to support the songwriting process by predicting chord progressions. Choosing effective chord progressions can be both a creative and technical challenge, as they establish mood, tension, and resolution in music. This tool assists songwriters by suggesting harmonically coherent chords based on prior context, helping to inspire new musical ideas. 

<img width="878" height="339" alt="Screenshot 2026-03-03 164949" src="https://github.com/user-attachments/assets/9efe8885-e1f5-4790-bd45-feccc11e131f" />

_We implemented 4 different models that might show 4 different suggestions_

It is particularly targeted toward beginners and amateur musicians who are navigating the basics of music harmony and may lack the experience to instinctively generate harmonic progressions. Rather than functioning as an autonomous complete music generator, The BeatlesBot is designed to provide a rich palette of possible creative choices.

<img width="861" height="442" alt="Screenshot 2026-03-03 165005" src="https://github.com/user-attachments/assets/3653be93-85e4-457f-b88f-9ef988f8cec4" />

_First step is to insert the first two chords of the sequence as a relative chord to the root note of the song.
Each chord is divided in Relative degree + color, witht this latter being the "attribute of the chord"_

<img width="1337" height="706" alt="Screenshot 2026-03-03 165049" src="https://github.com/user-attachments/assets/848e6ad0-c1c4-4b29-afb5-ad927562fd4c" />

_The green chords are the 4 most likely suggestions from the models, clicking on the chosen suggestion will generate 4 new possible chords, advancing of one step._

<img width="1335" height="695" alt="Screenshot 2026-03-03 165112" src="https://github.com/user-attachments/assets/94e0a9f9-dd69-4b4c-935f-5b7f5e1868d6" />

## Dataset and Preprocessing
* The models are trained on a comprehensive dataset of chords spanning the entire Beatles discography (Harte 2010).
* The original dataset can be found here: http://isophonics.net/datasets.
* A custom data-scraping pipeline was built to densify the data, expressing chords as relative degrees with respect to the root key of the song rather than as absolute values.
* Chords were divided into two core values: the **degree** (relating the chord to the root key, totaling 12 degrees) and the **color** (attributes like major, minor, minor 7th, augmented, etc.).
* To further densify the data, infrequently appearing colors were simplified to less expressive variations of the same chord.

## Implemented Models
Various algorithms were explored and evaluated, including both unsupervised learning methods and knowledge-based models incorporating music theory.

### 1. Bayesian Networks
* The Bayesian Network approach proved to be the most reliable method implemented.
* The final model utilizes a 2nd degree-depth architecture that considers the last two chords and their respective colors.
* It includes a specific network edge mapping the predicted chord degree to its color.
* To solve the issue of data sparsity—which initially caused flat likelihoods for unforeseen chord pairs—a custom inference function utilizing sampling was implemented.
* This sampling approach ensures the model returns non-flat distributions even when provided with unusual or unseen chord seeds.

### 2. Hidden Markov Models (HMM)
* This model was designed to yield predictions based on harmonic knowledge by using macro categories of chord degrees as hidden states.
* The network was initialized with a states-transition matrix and probability distributions for chords belonging to specific states.
* While the abstraction of chord functions during fitting did not perfectly match strict music theory, the model successfully suggests valid and musically pleasurable chord progressions.

### 3. Markov Chains
* Considered a simplified version of the Bayesian Network approach, this model merges chord degrees and colors into a single value.
* The model was fitted to the entire dataset using trigrams of chords to ensure more accurate sequence predictions.
* A custom `MarkovChain` class was developed specifically for this project since using integrated library functions proved overly difficult.

## Technology Stack
* **Language:** Fully developed in Python.
* **GUI:** Built using `custom tkinter` with assistance from code-generating online tools.
* **Libraries:** The `.pgmpy` library was used for the Bayesian Networks, and the `pomegranate` library was used for the Hidden Markov Models.

## Limitations and Observations
* The suggested chords across all models generally turn out to be musically pleasurable or interesting.
* Aside from the Bayesian Network with sampling, the other models heavily rely on previously seen sequences and struggle with robust predictions for unforeseen chord combinations.
* The current system does not track the time positioning of chords during training or inference.
* Consequently, while chord successions make sense from a local perspective, they may lack global structure and a broader harmonic narrative.

## Future Work
* Future iterations may explore the implementation of neural networks or higher-level HMMs.
* There is potential to develop a solution that better integrates explicit music theory rules with learned evidence.
