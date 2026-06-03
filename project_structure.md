# **Repository Structure for Deliverables**

As we progress in the course, organization is as important as the code you write. To ensure all projects are easy to review, understand, and replicate, we will use a standardized repository structure for each deliverable.

### The Golden Rule: Clarity and Reproducibility

Each of your deliverables will be contained in its own main folder (deliverable 1, deliverable 2, etc.), but inside them, you must always include the following organization.

By following this structure, we ensure that anyone (including yourself in the future) can:
- Understand the Project quickly by reading README.md.
- Run the Code smoothly thanks to requirements.txt.
- Locate the main source code inside the src/ folder.

### Detailed Structure per Deliverable

**IMPORTANT**: The following is a detailed suggestion; your structure does not need to have exactly the details shown below. The goal is to show the structure typically found in a project.

For each deliverable, your folder should replicate the following hierarchy (as far as possible), adapting internal files as necessary:

```
📦 project-name/
│
├── 📂 docs/                     # Project documentation
│   ├── 📜 README.md             # Extended documentation
│   ├── 📜 architecture.md       # Model details and UI design
│   ├── 📜 api.md               # API documentation (if applicable)
│   └── 📜 installation.md      # Installation guide and dependencies
│
├── 📂 src/                      # Main source code
│   ├── 📂 data/                 # Scripts to load/preprocess data
│   │   └── preprocess.py
│   ├── 📂 models/               # Model architecture definitions
│   │   └── my_model.py
│   ├── 📂 training/             # Training scripts
│   │   └── train.py
│   ├── 📂 evaluation/           # Validation and testing scripts
│   │   └── evaluate.py
│   ├── 📂 utils/                # Auxiliary functions
│   │   └── helpers.py
│   └── main.py                  # Main entry point (Streamlit)
│
├── 📂 notebooks/                # Jupyter Notebooks for experimentation
│   └── experiment_1.ipynb
│
├── 📂 experiments/              # Experiment results
│   ├── 📂 logs/                 # Training logs
│   ├── 📂 checkpoints/          # Saved model weights
│   └── 📂 results/              # Metrics, charts, outputs
│
├── 📂 tests/                    # Unit and integration tests
│   └── test_models.py
│
├── 📜 requirements.txt          # Python dependencies
├── 📜 environment.yml           # Alternative conda environment
├── 📜 .gitignore                # Git ignore configuration
├── 📜 LICENSE                   # License details
└── 📜 README.md                 # Main project description
```
