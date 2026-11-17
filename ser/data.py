"""Loading utilities for the pre-extracted IEMOCAP features and text CSV."""
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from ser.config import EMOTIONS, EXCLUDED_LABEL, FEATURES_PKL, TEXT_CSV


@dataclass
class IemocapFeatures:
    """Contents of IEMOCAP_features.pkl, keyed by dialogue ID.

    speakers  : per-utterance speaker ('M' / 'F')
    labels    : per-utterance integer emotion label
    text      : TextCNN text features (100-d)
    audio     : openSMILE acoustic features (100-d)
    visual    : 3D-CNN visual features (512-d)
    sentences : raw transcript of each utterance
    train_ids : dialogue IDs of the official training split (120)
    test_ids  : dialogue IDs of the official test split (31)
    """

    dialogue_ids: list
    speakers: dict
    labels: dict
    text: dict
    audio: dict
    visual: dict
    sentences: dict
    train_ids: list
    test_ids: list


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See the 'Data' section of README.md for download instructions."
        )
    return path


def load_features(path: Path = FEATURES_PKL) -> IemocapFeatures:
    """Load the pickled IEMOCAP feature dump (DialogueRNN format)."""
    with open(_require(path), "rb") as f:
        (ids, speakers, labels, text, audio, visual,
         sentences, train_ids, test_ids) = pickle.load(f, encoding="latin1")
    return IemocapFeatures(list(ids), speakers, labels, text, audio, visual,
                           sentences, list(train_ids), list(test_ids))


def utterances_dataframe(feats: IemocapFeatures) -> pd.DataFrame:
    """Flatten all dialogues into one row per utterance, dropping the excluded label."""
    rows = []
    for vid in feats.dialogue_ids:
        for speaker, label, text, audio, visual, sentence in zip(
            feats.speakers[vid], feats.labels[vid], feats.text[vid],
            feats.audio[vid], feats.visual[vid], feats.sentences[vid],
        ):
            if label == EXCLUDED_LABEL:
                continue
            rows.append({
                "VideoID": vid,
                "Speaker": speaker,
                "Label": label,
                "Text": text,
                "Audio": audio,
                "Visual": visual,
                "Sentence": sentence,
            })
    return pd.DataFrame(rows)


def audio_dataset(feats: IemocapFeatures, dialogue_ids) -> tuple[np.ndarray, list]:
    """Return (X, y) of openSMILE audio features and emotion names for the given dialogues."""
    wanted = set(dialogue_ids)
    X, y = [], []
    for vid in feats.dialogue_ids:
        if vid not in wanted:
            continue
        for label, audio in zip(feats.labels[vid], feats.audio[vid]):
            emotion = EMOTIONS.get(int(label))
            if emotion is None:  # excluded or unknown label
                continue
            X.append(audio)
            y.append(emotion)
    return np.asarray(X), y


def load_text_csv(path: Path = TEXT_CSV) -> pd.DataFrame:
    """Load the transcript CSV (columns: FileName, Sentences, Label, text, AugmentedText)."""
    return pd.read_csv(_require(path), index_col=0)
