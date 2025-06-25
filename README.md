# Spike-Walzen_Modell
"An efficient, hybrid language model based on the Spike-Walze architecture, trained with curriculum learning on German Wikipedia texts."

# Spike-Walze V3

An adaptive, resource-efficient language model architecture inspired by first-principles engineering.

The name *Spike-Walze* is German and translates to "Spike-Roller", reflecting its core architectural components that enable a dynamic, input-aware computational process.

## Core Concept: Intelligent Resource Allocation

Unlike standard Transformers that utilize their massive computational power for every token, Spike-Walze operates on a principle of adaptive, conditional computation. Each layer in the model features a dual-pathway system:

1.  **The Routine Pathway (`FastCircularWalze`):** An extremely fast and efficient convolutional mechanism. It excels at processing local patterns and standard sentence structures with minimal computational cost. This is the default path for the majority of tokens.
2.  **The Expert Pathway (`Local Attention`):** A precise, but more computationally expensive, local attention mechanism. It is similar to the attention in large models but is restricted to a local window to maintain efficiency.

The **`AdaptiveSpikeDetector`** acts as an intelligent gating mechanism. It analyzes the data flow in real-time and decides whether the efficient routine pathway is sufficient, or if the expert pathway needs to be activated ("spiked") for a more complex analysis.

## Key Innovations & Advantages

This model is a unique synthesis of several state-of-the-art concepts, leading to significant advantages:

* **Adaptive Computation:** The model's computational cost (FLOPs) scales with the input's difficulty. Simple sentences are processed with extreme efficiency, drastically lowering the average energy consumption per query.
* **Linear Time Complexity:** By combining a linear-time convolutional "roller" and local attention, the model avoids the quadratic ($O(n^2)$) bottleneck of standard Transformers. It operates in linear time ($O(n)$), enabling the efficient processing of very long contexts.
* **Emergent Division of Labor:** The architecture enables a natural separation of concerns, a property that emerges from its design:
    * The **`Walze`** handles efficient, *global* context propagation across layers (due to its circular nature).
    * The **`Local Attention`** acts as a high-precision tool for analyzing complex grammatical structures within a *local* window.
* **Structured Learning:** The model leverages a built-in Curriculum Learning system, allowing it to build robust knowledge by progressing from simple to complex examples, mirroring human learning patterns.

## How to Use

The entire training setup is self-contained in a single script.

### 1. Setup
Ensure you have Python 3 installed. Clone this repository and install the necessary dependencies using the following command:
```bash
pip install -r requirements.txt

2. Start Training

To start a new training run from scratch, simply execute the main script:
python SpikeWalze_V3_Sprachmodell_Wiki_Circulum_Training.py

3. Resume from a Checkpoint

To resume training from a saved checkpoint, edit the config dictionary at the bottom of the script:
config = {
    # ...
    # Set the path to your checkpoint file here
    'model_path': 'checkpoints/model_epoch_100.pt' 
}

The Origin Story

The core ideas for this model were inspired by a cross-disciplinary transfer of concepts from mechanical engineering and metal construction (Metallbau in German). Principles like structural integrity, process efficiency (e.g., using jigs and fixtures), and resource conservation were applied to the abstract world of neural networks.
