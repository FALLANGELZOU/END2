# END^2: Robust Dual-Decoder Watermarking Framework Against Non-Differentiable Distortions

## Overview
END^2 is a deep learning-based digital watermarking framework that enables end-to-end training against arbitrary non-differentiable distortions. It leverages self-supervised contrastive learning with a teacher-student architecture to achieve robust watermarking that can withstand various types of real-world noise and transformations.

The system uses a specialized encoder for message embedding and dual decoders (teacher and student) for extraction. The teacher network guides the student network through contrastive learning to handle non-differentiable distortions, enabling the framework to learn from operations where gradients cannot be directly computed.

## Usage
To train the model:
```
python main.py
```

## Configuration
It can be configured through the `cfg.yaml` file with parameters such as:
- Message length
- Image size
- Loss weights
- Training hyperparameters


## Project Structure
- `main.py`: Entry point for training
- `model.py`: Contains the END^2 model architecture
- `trainer.py`: Implementation of the training framework
- `T1Data.py`: Dataset loading and preprocessing
- `cfg.yaml`: Configuration parameters


