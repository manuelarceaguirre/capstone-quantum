```mermaid
flowchart TD

A[Load FRED-MD Dataset] --> B[Select Target Variables]

B --> C[Preprocessing Pipeline]

subgraph Q_pre["Preprocessing Question Block"]
C1[Handle Missing Data?]
C2[Apply PCA / Dimensionality Reduction?]
C3[Include COVID-19 Features?]
C4[Define Pseudo-Out-of-Sample Split?]
end

C --> C1 --> C2 --> C3 --> C4 --> D[Baseline Dataset]

subgraph E1["Classical Reservoir"]
E1a[Number of Layers]
E1b[Activation Functions]
E1c[Number of Random Initialization]
end

subgraph E[Quandum Implementation Questions]

subgraph E2[Quantum Reservoir]
E2a[Encoding Scheme]
E2b[Coherence]
E2c[Observables]
end

subgraph E3[Quantum Feature Map]
E3a[Encoding Scheme]
E3b[Observables]
end

Ea[Error Correction and Mitigation]
Eb[How to Estimate Observables]
end
D --> E1 --> E1a --> E1b --> E1c --> F
D --> E2a --> E2b --> E2c --> F
D --> E3a --> E3b --> F

F[Benchmarking Pipeline]

subgraph Q_models["Forecasting Model Block"]
F1[VARIMA]
F2[GLM]
F3[SVR]
F4[Tree-Based Models]
F5[TimeGPT]
end

F --> F1 --> G[Evaluation Stage]
F --> F2 --> G
F --> F3 --> G
F --> F4 --> G
F --> F5 --> G

subgraph Q_eval["Evaluation Question Block"]
G1[RMSE]
G2[MAE]
G3[MAPE]
G4[RMSPE]
end

G --> G1
G --> G2
G --> G3
G --> G4

G1 --> H[Representation Comparison]
G2 --> H[Representation Comparison]
G3 --> H[Representation Comparison]
G4 --> H[Representation Comparison]
H --> I[Best Method Selection]
I --> J[Analysis & Conclusions]
