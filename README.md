# PyTorch Transformer for Machine Translation
# Transformer Architecture: PyTorch Implementation

This project is a complete implementation of the Transformer architecture from the seminal paper _"Attention Is All You Need"_ by Vaswani et al. The model is built from scratch using PyTorch and is trained on the Multi30k dataset for a German-to-English machine translation task.

The implementation is based on the Harvard NLP group's annotated guide and was completed for the "PA2: The Transformer Architecture" assignment in the DS 5983 course.

The project not only builds and trains the model but also includes a systematic framework for hyperparameter experimentation and analysis, allowing for the comparison of different model configurations.


---

## Features

- **Full Transformer Implementation:** Complete Encoder-Decoder architecture built with modular PyTorch `nn.Module` classes.
- **Core Components from Scratch:** Includes implementations of Multi-Head Self-Attention, Position-wise Feed-Forward Networks, and Sinusoidal Positional Encodings.
- **Machine Translation Task:** Trained and evaluated on the Multi30k German-to-English dataset.
- **Greedy Decoding Inference:** A `translate_sentence` function to perform inference on new sentences using a greedy decoding strategy.
- **Systematic Hyperparameter Tuning:** A robust framework to automatically run experiments with different combinations of hyperparameters (`num_heads`, `num_layers`, `learning_rate`, `batch_size`).
- **Performance Analysis:** Uses pandas and matplotlib to analyze and visualize the results of hyperparameter experiments to find the optimal configuration.

---

## Architecture Overview

The model is composed of several key components, each implemented as a separate class for modularity:

- **Multi-Head Attention (`MultiHeadAttention`):** Implements the scaled dot-product attention mechanism, split across multiple heads to allow the model to jointly attend to information from different representation subspaces.
- **Position-wise Feed-Forward Network (`PositionwiseFeedForward`):** A simple, fully connected feed-forward network applied to each position separately and identically.
- **Positional Encoding (`PositionalEncoding`):** Since the model contains no recurrence, positional encodings are added to the input embeddings to give the model information about the relative or absolute position of tokens.
- **Encoder Layer (`EncoderLayer`):** Consists of a multi-head self-attention sublayer and a feed-forward sublayer.
- **Decoder Layer (`DecoderLayer`):** Consists of a masked multi-head self-attention sublayer, a cross-attention sublayer (attending to the encoder's output), and a feed-forward sublayer.
- **Full Transformer (`Transformer`):** Stacks the encoder and decoder layers, manages embeddings, and includes the final linear layer to produce output probabilities. It also handles the creation of padding masks and look-ahead masks.

