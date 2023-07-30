import torch
from torch import nn
from speechbrain.pretrained import EncoderClassifier
from speechbrain.processing.features import Deltas, ContextWindow

def normalize_tensor(input_tensor):
    mean = torch.mean(input_tensor, dim=1)[:, None]
    normalized_tensor = input_tensor - mean
    return normalized_tensor

class SpeakerEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="spkrec-ecapa-voxceleb"
        )
        classifier.eval()

        self.embedding_model = classifier.mods.embedding_model

    def forward(self, x):
        lens = torch.ones(x.size(0))
        x = normalize_tensor(x)
        x = self.embedding_model(x, lens)
        return x

x = torch.rand(1, 101, 80)
model = SpeakerEmbeddings()
traced_script_module = torch.jit.trace(model, x)
traced_script_module.save("speaker-embeddings.pt")
