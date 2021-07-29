
<p  align="center"><img src="https://user-images.githubusercontent.com/42150335/127541215-931f2271-5c17-4672-a328-c8fafc4a8da9.png" height=100>
  

<div align="center">

**Unofficial PyTorch implementation of [Luna: Linear Unified Nested Attention](https://arxiv.org/pdf/2106.01540.pdf)**

  
</div>
  
***
  
The quadratic computational and memory complexities of the Transformer’s attention mechanism have limited its scalability for modeling long sequences. In
this paper, we propose Luna, a linear unified nested attention mechanism that
approximates softmax attention with two nested linear attention functions, yielding
only linear (as opposed to quadratic) time and space complexity. As compared to
a more traditional attention mechanism, Luna introduces an additional sequence
with a fixed length as input and an additional corresponding output, which allows
Luna to perform attention operation linearly, while also storing adequate contextual
information. We perform extensive evaluations on three benchmarks of sequence
modeling tasks: long-context sequence modeling, neural machine translation and
masked language modeling for large-scale pretraining. Competitive or even better
experimental results demonstrate both the effectiveness and efficiency of Luna
compared to a variety of strong baseline methods including the full-rank attention
and other efficient sparse and dense attention methods

## Installation
This project recommends Python 3.7 or higher.
We recommend creating a new virtual environment for this project (using virtual env or conda).
  
### Prerequisites
* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* Pytorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.  
  
### Install from source
Currently we only support installation from source code using setuptools. Checkout the source code and run the
following commands:  
  
```
pip install -e .
```

## Usage

```python
import torch
from luna_transformer import LunaTransformerEncoder

DUMMY_INPUTS = torch.LongTensor([
    [2, 3, 3, 3, 3, 3, 2, 2, 0],
    [2, 3, 3, 3, 3, 3, 2, 3, 2],
    [2, 3, 3, 3, 3, 3, 2, 2, 0],
])
DUMMY_INPUT_LENGTHS = torch.IntTensor([9, 8, 7])

model = LunaTransformerEncoder(vocab_size=4, d_model=512, num_layers=6,
                               num_attention_heads=8, project_embedding_length=32,
                               dropout_p=0.1, max_length=1024)
ouputs = model(DUMMY_INPUTS, DUMMY_INPUT_LENGTHS)
```

## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/conformer/issues) on github or   
contacts sh951011@gmail.com please.
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
## Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.

## Author
  
* Soohwan Kim [@sooftware](https://github.com/sooftware)
* Contacts: sh951011@gmail.com
