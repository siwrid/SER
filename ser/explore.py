"""Flatten IEMOCAP_features.pkl into a per-utterance DataFrame and summarise it.

Usage:
    python -m ser.explore [--out data/utterances.pkl]
"""
import argparse

from ser.config import EMOTIONS
from ser.data import load_features, utterances_dataframe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="optional path to save the DataFrame (pickle)")
    args = parser.parse_args(argv)

    feats = load_features()
    df = utterances_dataframe(feats)
    print(df[["VideoID", "Speaker", "Label", "Sentence"]])
    print(f"\n{len(df)} utterances across {df['VideoID'].nunique()} dialogues")
    print("\nLabel distribution:")
    print(df["Label"].map(EMOTIONS).value_counts().to_string())

    if args.out:
        df.to_pickle(args.out)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
