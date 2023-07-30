import torch
from torch import nn
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer

class WhisperLikelihoodEstimator(nn.Module):
    def __init__(self, model_name: str, decoder_prefix: str, decoder_candidate: str):
        super().__init__()

        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.whisper_model.eval()

        self.whisper_encoder = self.whisper_model.get_encoder()
        self.whisper_decoder = self.whisper_model.get_decoder()

        self.whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_name
        )

        tokenizer = WhisperTokenizer.from_pretrained(model_name)
        self.decoder_prefix = tokenizer(decoder_prefix, return_tensors="pt")["input_ids"]
        self.decoder_candidate = tokenizer(decoder_candidate, add_special_tokens=False, return_tensors="pt")["input_ids"]

    def forward(
        self,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        decoder_input_ids = torch.cat(
            [self.decoder_prefix, self.decoder_candidate], dim=-1
        )

        encoded_input = self.whisper_encoder(audio_features).last_hidden_state
        decoder_states = self.whisper_decoder(
            decoder_input_ids, encoder_hidden_states=encoded_input
        ).last_hidden_state
        decoder_logits = self.whisper_model.proj_out(decoder_states)
        decoder_logits = torch.softmax(decoder_logits, dim=-1)

        decoder_logits = decoder_logits[0, :-1, :]
        decoder_logits = decoder_logits[-len(self.decoder_candidate[0]) :, :]
        probabilities = torch.stack([
            decoder_logits[ts_idx, class_idx]
            for (ts_idx, class_idx) in enumerate(self.decoder_candidate[0])
        ])

        likelihood = torch.prod(probabilities)
        return likelihood

def main():
    model = WhisperLikelihoodEstimator(
        model_name="openai/whisper-tiny.en",
        decoder_prefix=" (Noise)",
        decoder_candidate=" Hey Pepper",
    )

    audio = torch.randn(1, 80, 3000)

    traced_model = torch.jit.trace(model, audio)
    traced_model = torch.jit.optimize_for_inference(traced_model)
    traced_model.save("whisper-likelihood.pt")

if __name__ == "__main__":
    main()
