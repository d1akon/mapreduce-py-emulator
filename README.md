# mapreduce-py-emulator

This repo contains a Python-based implementation of a MapReduce framework along with an example job that computes Term Frequency-Inverse Document Frequency (TF-IDF) for a collection of text reviews. The MapReduce framework (`MRE.py`) simulates the core functionalities of the MapReduce paradigm, enabling parallel processing of large datasets. The TF-IDF job (`tfidf.py`) utilizes this framework to analyze review data, calculating the importance of terms within each document relative to the entire corpus.

## Features

- **Custom MapReduce Framework:** A Python implementation that mimics the MapReduce process, including mapping, shuffling, sorting, and reducing phases.
- **TF-IDF Calculation:** An example job that computes TF-IDF scores for words in review documents, identifying the most significant terms.
- **Flexible Configuration:** Easily configurable number of mappers and reducers, with support for combiners and custom comparison functions.
- **Modular Design:** Clear separation between the MapReduce framework and the specific TF-IDF job, facilitating reuse and extension.
