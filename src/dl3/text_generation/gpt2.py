import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class GPT2(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="dl3",
    repo_url="https://github.com/not-lain/dl3",
    tags=["gpt2", "text-generation"],
):
    def __init__(self, a, b):
        super().__init__()
        self.l = nn.Linear(a, b)


if __name__ == "__main__":
    model = GPT2(10, 10)
    print(model)
