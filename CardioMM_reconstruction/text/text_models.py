"""
Created on 2025/09/05
@author: Zi Wang
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.
"""


import torch
from torch import nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer, AutoModel, AutoTokenizer
import os

# Models that use mean pooling
POOL_MODELS = {"sentence-transformers/all-MiniLM-L6-v2", "TaylorAI/bge-micro-v2", "../bge-micro-v2"}


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class LanguageModel(nn.Module):
    def __init__(self, llm_model_name='distilbert-base-uncased'):
        super(LanguageModel, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModel.from_pretrained(llm_model_name)
        self.model_name = llm_model_name
        # Remove the CLIP vision tower
        if "clip" in self.model_name:
            self.model.vision_model = None
        # Freeze the pre-trained parameters (very important)
        for param in self.model.parameters():
            param.requires_grad = False

        # Make sure to set evaluation mode (also important)
        self.model.eval()

    def forward(self, text_batch):
        inputs = self.tokenizer(text_batch, padding=True, max_length=512, truncation=True, return_tensors="pt")
        with torch.no_grad():  # Ensure no gradients are computed for this forward pass

            if "clip" in self.model_name:
                sentence_embedding = self.model.get_text_features(**inputs)
                return sentence_embedding

            outputs = self.model(**inputs)

        if any(model in self.model_name for model in POOL_MODELS):
            sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
            # Normalize embeddings
            sentence_embedding = F.normalize(sentence_embeddings, p=2, dim=1)
        else:
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
        return sentence_embedding


class LanguageModel_NoAutoTokenizer(nn.Module):
    def __init__(self, llm_model_name='distilbert-base-uncased'):
        super(LanguageModel_NoAutoTokenizer, self).__init__()

        self.model = AutoModel.from_pretrained(llm_model_name)
        self.model_name = llm_model_name
        # Remove the CLIP vision tower
        if "clip" in self.model_name:
            self.model.vision_model = None
        # Freeze the pre-trained parameters (very important)
        for param in self.model.parameters():
            param.requires_grad = False

        # Make sure to set evaluation mode (also important)
        self.model.eval()

    def forward(self, inputs):
        with torch.no_grad():  # Ensure no gradients are computed for this forward pass

            if "clip" in self.model_name:
                sentence_embedding = self.model.get_text_features(**inputs)
                return sentence_embedding

            outputs = self.model(**inputs)

        if any(model in self.model_name for model in POOL_MODELS):
            sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
            # Normalize embeddings
            sentence_embedding = F.normalize(sentence_embeddings, p=2, dim=1)
        else:
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
        return sentence_embedding
    

class LMHead(nn.Module):
    def __init__(self, llm_model_dim=384, llm_embd_dim=256, llm_nclasses=3):
        super(LMHead, self).__init__()
        
        self.fc1 = nn.Linear(llm_model_dim, llm_embd_dim)
        # self.gelu = nn.GELU()
        self.fc2 = nn.Linear(llm_embd_dim, llm_nclasses)
        
    def forward(self, x):
        embd = self.fc1(x)
        embd = F.normalize(embd, p=2, dim=1)
        # no use in our reconstruction task, or can be equalled to the number of undersampling patterns
        deg_pred = self.fc2(embd)
        return embd, deg_pred


class LMHead2(nn.Module):
    def __init__(self, llm_model_dim=384, llm_embd_dim=256, llm_nclasses=3):
        super(LMHead2, self).__init__()

        self.fc1 = nn.Linear(llm_model_dim, llm_embd_dim)
        # self.gelu = nn.GELU()
        self.fc2 = nn.Linear(llm_embd_dim, llm_nclasses)

    def forward(self, x):
        embd = self.fc1(x)
        embd = F.normalize(embd, p=2, dim=1)
        # no use in our reconstruction task, or can be equalled to the number of undersampling patterns
        deg_pred = self.fc2(embd)
        return embd, deg_pred