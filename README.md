ENGINE_ALPHA/
│
├── config/
│   ├── global_config.yaml         # Central repository for hyperparameters, paths, and flags
│   └── feature_config.py          # Strict definitions of input features vs target columns
│
├── database/                      # Persistent storage for scraped or streamed live tables
│   └── wingo_history.db           # SQLite database for keeping live history intact
│
├── datasets/                      # Cleaned data ready for model ingestion
│   └── WinGo30S_Ready_data.csv    # Output from our preprocessing pipeline
│
├── artifacts/                     # Serialized model weights, scalers, and metadata
│   ├── scalers/                   # Saved MinMaxScaler/StandardScaler parameters
│   ├── supervised/                # Saved weights for LSTM, Transformer, XGBoost, LightGBM
│   ├── unsupervised/              # Saved Autoencoder weights (.pt)
│   ├── reinforcement/             # Saved DQN policy networks
│   └── meta_learner/              # Saved weights for the complex mathematical aggregator
│
├── logs/                          # System performance tracks and error logging
│   ├── training.log               # Logs performance metrics (Loss, Accuracy, Val metrics)
│   └── live_inference.log         # Track live performance and API latencies
│
├── extras/                        # Scrapers, legacy scripts, and raw web captures
│   ├── web_scraper.py             # Script to grab real-time results from the platform
│   └── WinGo30S_fetched_pages.csv # Your original 900k-row raw dataset
│
└── src/                           # Core application source code
    ├── __init__.py
    │
    ├── data/                      # Data pipeline execution
    │   ├── __init__.py
    │   ├── preprocessor.py        # Clean, engineer, and transform tabular features
    │   └── dataset_loader.py      # Custom PyTorch Dataset class for sequence generation
    │
    ├── models/                    # Pure model architecture classes (No training logic here)
    │   ├── __init__.py
    │   ├── lstm_brain.py          # PyTorch LSTM architecture definition
    │   ├── transformer_brain.py   # PyTorch Time-Series Attention-head architecture
    │   ├── autoencoder.py         # PyTorch Anomaly Detector network
    │   └── dqn_agent.py           # Reinforcement Learning policy and replay buffer logic
    │
    ├── training/                  # Dedicated isolated training engines
    │   ├── __init__.py
    │   ├── train_tabular.py       # Trains and evaluates XGBoost and LightGBM
    │   ├── train_sequences.py     # Deep learning training loop (LSTM & Transformer)
    │   ├── train_autoencoder.py   # Self-supervised learning loop for anomaly capture
    │   └── train_meta_learner.py  # Out-of-fold stacking script to train the aggregator
    │
    ├── ensemble/                  # The complex mathematical aggregation layer
    │   ├── __init__.py
    │   └── meta_aggregator.py     # Math equations / neural stacker to merge model outputs
    │
    └── inference/                 # Real-time orchestration layer
        ├── __init__.py
        └── live_engine.py         # Main execution script running the 30-second live loop








┌────────────────────────────────────────────────────────┐
  │                                                        │
  ▼                                                        │
[Kaggle Cloud (Online)]                                    │
  └── 1. Upload new CSV logs                               │
  └── 2. Train Models (XGBoost, LSTM, Transformer)         │
  └── 3. Download trained weights (*.pt, *.json)           │
                                                           │
[Local Machine (Offline)]                                  │
  └── 4. Drop weights into artifacts/ folder               │
  └── 5. Live Engine captures game states locally          │
  └── 6. Real-time inference executed locally             │
  └── 7. Logs appends to local wingo_history.db ───────────┘





