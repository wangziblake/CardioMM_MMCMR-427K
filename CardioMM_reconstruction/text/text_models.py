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
# These sentence-transformer style encoders are intended to use masked mean
# pooling rather than the first-token/CLS representation.
POOL_MODELS = {"sentence-transformers/all-MiniLM-L6-v2", "TaylorAI/bge-micro-v2", "../bge-micro-v2"}


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    """
    Compute attention-mask-aware mean pooling over token embeddings.

    Args:
        model_output: Hugging Face model output whose first item is the token
            embedding tensor shaped ``[batch, sequence_length, hidden_dim]``.
        attention_mask: Token mask shaped ``[batch, sequence_length]`` where
            non-padding tokens are 1 and padding tokens are 0.

    Returns:
        Sentence embeddings shaped ``[batch, hidden_dim]``.
    """
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    # Expand the mask over the hidden dimension so padding tokens contribute zero.
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class LanguageModel(nn.Module):
    """
    Frozen text encoder wrapper that owns its tokenizer.

    This module accepts raw text strings, tokenizes them internally, and returns
    one sentence embedding per input. It is useful for standalone text encoding
    outside data pipelines that already produce tokenized Hugging Face inputs.
    """

    def __init__(self, llm_model_name='distilbert-base-uncased'):
        """
        Args:
            llm_model_name: Hugging Face model name or local path used for both
                tokenizer and encoder weights.
        """
        super(LanguageModel, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModel.from_pretrained(llm_model_name)
        self.model_name = llm_model_name
        # Remove the CLIP vision tower because this wrapper only uses text features.
        if "clip" in self.model_name:
            self.model.vision_model = None
        # Freeze the pre-trained parameters so reconstruction training does not update the encoder.
        for param in self.model.parameters():
            param.requires_grad = False

        # Keep dropout and other training-time layers disabled for deterministic embeddings.
        self.model.eval()

    def forward(self, text_batch):
        """
        Encode raw text strings into sentence embeddings.

        Args:
            text_batch: String or list of strings to tokenize with padding,
                truncation, and a maximum length of 512 tokens.

        Returns:
            Tensor of sentence embeddings shaped ``[batch, hidden_dim]``. CLIP
            models use ``get_text_features``; configured pooling models use
            masked mean pooling and L2 normalization; other models use the first
            token representation.
        """
        inputs = self.tokenizer(text_batch, padding=True, max_length=512, truncation=True, return_tensors="pt")
        with torch.no_grad():  # Ensure no gradients are computed for this forward pass

            if "clip" in self.model_name:
                # CLIP exposes a dedicated text-feature projection.
                sentence_embedding = self.model.get_text_features(**inputs)
                return sentence_embedding

            outputs = self.model(**inputs)

        if any(model in self.model_name for model in POOL_MODELS):
            sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
            # Unit-length embeddings are used as stable conditioning vectors.
            sentence_embedding = F.normalize(sentence_embeddings, p=2, dim=1)
        else:
            # BERT-like encoders commonly use the first token as sentence embedding.
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
        return sentence_embedding


class LanguageModel_NoAutoTokenizer(nn.Module):
    """
    Frozen text encoder wrapper for pre-tokenized Hugging Face inputs.

    This is the variant used by the CardioMM training module, where metadata and
    undersampling descriptions are tokenized by the data pipeline and passed as
    dictionaries containing keys such as ``input_ids`` and ``attention_mask``.
    """

    def __init__(self, llm_model_name='distilbert-base-uncased'):
        """
        Args:
            llm_model_name: Hugging Face model name or local path for encoder
                weights. A tokenizer is intentionally not created in this class.
        """
        super(LanguageModel_NoAutoTokenizer, self).__init__()

        self.model = AutoModel.from_pretrained(llm_model_name)
        self.model_name = llm_model_name
        # Remove the CLIP vision tower because only text features are needed.
        if "clip" in self.model_name:
            self.model.vision_model = None
        # Freeze the pre-trained parameters so only reconstruction/projector layers train.
        for param in self.model.parameters():
            param.requires_grad = False

        # Keep the frozen encoder in inference mode during reconstruction training.
        self.model.eval()

    def forward(self, inputs):
        """
        Encode pre-tokenized text inputs into sentence embeddings.

        Args:
            inputs: Hugging Face tokenizer output dictionary, typically including
                ``input_ids``, ``attention_mask``, and optionally
                ``token_type_ids``.

        Returns:
            Tensor of sentence embeddings shaped ``[batch, hidden_dim]`` using
            the same CLIP, mean-pooling, or first-token strategy as
            ``LanguageModel``.
        """
        with torch.no_grad():  # Ensure no gradients are computed for this forward pass

            if "clip" in self.model_name:
                # CLIP exposes a dedicated text-feature projection.
                sentence_embedding = self.model.get_text_features(**inputs)
                return sentence_embedding

            outputs = self.model(**inputs)

        if any(model in self.model_name for model in POOL_MODELS):
            sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
            # Unit-length embeddings are used as stable conditioning vectors.
            sentence_embedding = F.normalize(sentence_embeddings, p=2, dim=1)
        else:
            # BERT-like encoders commonly use the first token as sentence embedding.
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
        return sentence_embedding
    

class LMHead(nn.Module):
    """
    Projection head for metadata text embeddings.

    The head adapts frozen language-model embeddings to the lower-dimensional
    embedding used by the MRI reconstruction network and also produces auxiliary
    class logits.
    """

    def __init__(self, llm_model_dim=384, llm_embd_dim=256, llm_nclasses=3):
        """
        Args:
            llm_model_dim: Input dimension of the frozen language-model embedding.
            llm_embd_dim: Output embedding dimension used for reconstruction
                conditioning.
            llm_nclasses: Number of auxiliary prediction classes.
        """
        super(LMHead, self).__init__()
        
        self.fc1 = nn.Linear(llm_model_dim, llm_embd_dim)
        # self.gelu = nn.GELU()
        self.fc2 = nn.Linear(llm_embd_dim, llm_nclasses)
        
    def forward(self, x):
        """
        Project and normalize language embeddings, then predict auxiliary logits.

        Args:
            x: Language-model embedding tensor shaped ``[batch, llm_model_dim]``.

        Returns:
            Tuple ``(embd, deg_pred)`` where ``embd`` is L2-normalized and shaped
            ``[batch, llm_embd_dim]``, and ``deg_pred`` is shaped
            ``[batch, llm_nclasses]``.
        """
        embd = self.fc1(x)
        # Normalize before passing embeddings into the reconstruction network.
        embd = F.normalize(embd, p=2, dim=1)
        # no use in our reconstruction task, or can be equalled to the number of undersampling patterns
        deg_pred = self.fc2(embd)
        return embd, deg_pred


class LMHead2(nn.Module):
    """
    Projection head for undersampling-description text embeddings.

    This mirrors ``LMHead`` but is instantiated separately so metadata and
    undersampling prompts can have independent projection parameters.
    """

    def __init__(self, llm_model_dim=384, llm_embd_dim=256, llm_nclasses=3):
        """
        Args:
            llm_model_dim: Input dimension of the frozen language-model embedding.
            llm_embd_dim: Output embedding dimension used for reconstruction
                conditioning.
            llm_nclasses: Number of auxiliary prediction classes.
        """
        super(LMHead2, self).__init__()

        self.fc1 = nn.Linear(llm_model_dim, llm_embd_dim)
        # self.gelu = nn.GELU()
        self.fc2 = nn.Linear(llm_embd_dim, llm_nclasses)

    def forward(self, x):
        """
        Project and normalize language embeddings, then predict auxiliary logits.

        Args:
            x: Language-model embedding tensor shaped ``[batch, llm_model_dim]``.

        Returns:
            Tuple ``(embd, deg_pred)`` where ``embd`` is L2-normalized and shaped
            ``[batch, llm_embd_dim]``, and ``deg_pred`` is shaped
            ``[batch, llm_nclasses]``.
        """
        embd = self.fc1(x)
        # Normalize before passing embeddings into the reconstruction network.
        embd = F.normalize(embd, p=2, dim=1)
        # no use in our reconstruction task, or can be equalled to the number of undersampling patterns
        deg_pred = self.fc2(embd)
        return embd, deg_pred
