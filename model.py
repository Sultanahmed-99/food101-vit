import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch import nn
import os 
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch the token from environment variable
access_token = os.environ.get('HF_TOKEN')


def create_image_classification_model(num_classes: int = 3, seed: int = 42):
    """Creates an image classification model using a fine-tuned transformer model.

    Args:
        num_classes (int, optional): Number of classes in the classifier head. Defaults to 3.
        seed (int, optional): Random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): Fine-tuned image classification model.
        processor (transformers.AutoImageProcessor): Image processor.
    """
    # Load the fine-tuned model and processor
    model_name = "ateraw/food"
    processor = AutoImageProcessor.from_pretrained(model_name , auth_token=access_token)
    model = AutoModelForImageClassification.from_pretrained(model_name) 

    # Freeze all layers in the base model
    for param in model.parameters():
        param.requires_grad = False

    # Change the classifier head with a random seed for reproducibility
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=model.classifier.in_features, out_features=num_classes),
    )

    return model, processor

 
