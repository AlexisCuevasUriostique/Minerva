# HegelianBERT: Clinical Sublation & Reasoning Trace

This repository contains the implementation of **HegelianBERT**, a specialized BERT-based model designed for clinical diagnostic reasoning. It introduces a novel **Differential Attention** mechanism to identify both supporting evidence (thesis) and contradictions (negation) within clinical texts, enabling a more nuanced diagnostic outcome.

## Project Structure

- `Hegelian_Bert.py`: The main Python script containing the model definition, weight download function, and a demonstration of its clinical diagnostic reasoning capabilities.
- `Hegelian_Bert.pth`: (Downloaded automatically by the script) Pre-trained model weights for HegelianBERT.

## Features

- **HegelianBERT Model**: Extends `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` with custom layers.
- **Differential Attention**: A unique attention mechanism that computes both 'positive' (supportive) and 'negative' (contradictory) attention weights, inspired by Hegelian dialectics.
- **Clinical Diagnostic Demo**: Showcases the model's ability to analyze clinical cases and provide a 'reasoning trace' with 'Thesis' and 'Negation' scores for diagnostic hypotheses.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. You will also need `pip` for dependency management.

### Installation

1.  **Clone the repository** (if this were a git repo):

    ```bash
    git clone https://github.com/your-username/HegelianBERT.git
    cd HegelianBERT
    ```

2.  **Install the required Python packages**:

    ```bash
    pip install torch transformers datasets requests
    ```

### Running the Demo

The `Hegelian_Bert.py` script will automatically download the necessary model weights and run a clinical diagnostic demo using a random case from the `shzyk/DiagnosisArena` dataset.

To run the demo:

```bash
python Hegelian_Bert.py
```

### Expected Output

The script will output a reasoning trace for a randomly selected clinical case, presenting diagnostic hypotheses with their 'Thesis (%)' (viability score) and 'Negation (%)' (contradiction score), and a final 'SUBLATED' or 'REJECTED' status. An asterisk `*` denotes the correct diagnosis.

```
=====================================================================================
 FELLOWSHIP DEMO: CLINICAL SUBLATION & REASONING TRACE 
=====================================================================================
CASE EXCERPT: A 12-year-old female presented with bilateral vision loss for 2 weeks after 5-month application of repeated low-level red-light (RLRL) laser exposure for bilateral moderate myopia. One month before presentation, the patient complained of abnormally bright light and prolonged afterimages after exposu...
-------------------------------------------------------------------------------------
DIAGNOSTIC HYPOTHESIS               | THESIS (%)   | NEGATION (%) | RESULT
-------------------------------------------------------------------------------------
Toxic retinopathy due to red-light  |      58.82% |      41.18% | SUBLATED  
Bilateral foveal photoreceptor and  |      46.79% |      53.21% | REJECTED *
Photic retinopathy secondary to low |      55.00% |      45.00% | SUBLATED  
Solar retinopathy-like phototoxicit |      34.08% |      65.92% | REJECTED  
-------------------------------------------------------------------------------------
PROPOSAL NOTE: The 'Negation' score quantifies clinical contradictions identified by
the Differential Attention layer, mirroring the Hegelian 'Antithesis'.
=====================================================================================
```

## Model Architecture

### HegelianBERT

```python
class HegelianBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        self.logic_layer = DifferentialAttention(768)
        self.norm = nn.LayerNorm(768)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, mask):
        x = self.bert(input_ids, mask).last_hidden_state
        return self.classifier(self.norm(x + self.logic_layer(x)).mean(dim=1))
```

### DifferentialAttention

```python
class DifferentialAttention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.scale = (h//8)**-0.5
        self.pos_q=nn.Linear(h,h); self.pos_k=nn.Linear(h,h); self.pos_v=nn.Linear(h,h)
        self.neg_q=nn.Linear(h,h); self.neg_k=nn.Linear(h,h)
        self.gate=nn.Linear(h*2,h); self.out=nn.Linear(h,h)
    def forward(self, x):
        p = torch.matmul(self.pos_q(x), self.pos_k(x).transpose(-2,-1))*self.scale
        p = F.softmax(p,-1)
        p = torch.matmul(p,self.pos_v(x))

        n = torch.matmul(self.neg_q(x), self.neg_k(x).transpose(-2,-1))*self.scale
        n = torch.sigmoid(n)
        n = torch.matmul(n,self.pos_v(x))

        g = torch.sigmoid(self.gate(torch.cat([p,n],-1)))
        return self.out(p*g - (n*(1-g)))
```
