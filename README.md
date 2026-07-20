# C-(a-i)-t

🐱 Cat Breed Classifier

A deep learning model that classifies cat images into their breed using transfer learning. Built on a pretrained EfficientNet-B0 backbone with a custom classification head (frozen feature extractor + dropout + linear layer), fine-tuned on the Cat Breeds Dataset (Kaggle). Trained with label smoothing, learning rate decay, and early stopping to handle class imbalance across dozens of breeds. Deployed as an interactive web app with Streamlit — upload a photo and get the top-3 predicted breeds with confidence scores.

Tech stack: PyTorch, torchvision, Streamlit