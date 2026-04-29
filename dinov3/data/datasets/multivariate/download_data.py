from aeon.datasets import load_classification
import os

# These are multivariate datasets from the UCR/UEA collection, grouped by domain.
# We exclude a few very small ones (e.g., BasicMotions, ERing, StandWalkJump)
# to focus on larger, more informative datasets for DINO pretraining.
multivariate_list = [
    # Motion / activity / gestures
    "NATOPS",                  # Motion capture: 24 channels (body sensors) - aircraft handling gestures
    "RacketSports",            # Motion capture: body-worn sensors during racket sports
    "UWaveGestureLibrary",     # Accelerometer-based hand gestures

    # Handwriting / trajectories
    "CharacterTrajectories",   # Pen-tip x/y + pressure over time
    "Handwriting",             # Online handwriting signals (pen trajectories)

    # EEG / physiology / neuro
    "FingerMovements",         # EEG: 28/56 channels - finger movement intention
    "SelfRegulationSCP1",      # EEG: slow cortical potentials - self-regulation task 1
    "SelfRegulationSCP2",      # EEG: slow cortical potentials - self-regulation task 2

    # Other rich multivariate sensor datasets
    "Epilepsy",                # 3-channel accelerometer - epileptic seizure detection
    "PEMS-SF",                 # Traffic flow sensors (many channels, long sequences)
    "ArticularyWordRecognition",  # Articulatory features while speaking words
    "DuckDuckGeese",           # Animal movement / trajectory-style multivariate signals
    "Heartbeat",               # Multichannel heartbeat-related time series
]

os.makedirs("data/multivariate", exist_ok=True)

for name in multivariate_list:
    print(f"Downloading {name}...")
    try:
        # load_classification returns X (the signals) and y (the labels)
        X, y = load_classification(name, extract_path="./data/multivariate")
        print(f"Loaded {name}: {X.shape} (Samples, Channels, Length)")
    except Exception as e:
        print(f"Error fetching {name}: {e}")