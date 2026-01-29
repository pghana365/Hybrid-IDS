# Hybrid-IDS

# Automotive Intrusion Detection System (IDS) using Generative AI
# Solving the Data Imbalance Problem in CAN Bus Security

## What is this?
This is the code repository for my research project on **Automotive Cybersecurity**.

One of the biggest problems in building AI for self-driving cars is that real attack data is extremely rare. You might have hours of normal driving logs, but only a few seconds of an attack. This "Class Imbalance" (often 20:1 or worse) makes standard AI models fail—they just guess "Normal" every time and miss the attacks.

My project solves this by using a Variational Autoencoder (VAE). Instead of just copying the attack data, the VAE learns the "recipe" of an attack and generates thousands of new, realistic synthetic attacks to balance the dataset 1:1.

##  Project Structure
There are two main phases as of now to this project, broken down into two notebooks:

### 1. 01_Data_Preprocessing.ipynb (Phase 1)
This script handles the raw data. It takes the messy CAN bus logs and turns them into something an AI can read.
* Input: Raw CSV/Log files (ROAD Dataset).
* What it does: Extracts features (CAN ID, Payload), converts Hex to Integers, and scales numbers between 0 and 1.
* Output: Saves a clean `.npz` file ready for training.

### 2. 02_VAE_Augmentation.ipynb (Phase 2)
* Training: It isolates the rare "Attack" messages and trains a VAE to understand their probability distribution.
* Generation: It generates enough synthetic attacks to exactly match the count of normal messages.
* Result: A perfectly balanced dataset (50% Normal, 50% Attack) saved as `balanced_dataset.npz`.

