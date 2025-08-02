# Troubleshooting Guide

This document helps resolve common issues with the Simple Decision AI system.

## Common Issues

### 1. Model Fails to Load
- **Symptom**: Error message "Failed to load model".
- **Solution**: Ensure the model path in `config/model_config.yaml` is correct and the model files are not corrupted.

### 2. Low Accuracy
- **Symptom**: The AI provides incorrect answers frequently.
- **Solution**: The model needs fine-tuning. Use the `debug` mode to collect feedback and then run the training script.
...
