"""Train an MLP emotion classifier on openSMILE acoustic features.

Usage:
    python -m ser.train_audio [--test-size 0.25] [--seed 9] [--no-plot]
"""
import argparse

import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from ser.config import FIGURES_DIR
from ser.data import audio_dataset, load_features


def build_model(seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(300,),
        alpha=0.01,
        batch_size=256,
        epsilon=1e-08,
        learning_rate="adaptive",
        max_iter=500,
        random_state=seed,
    )


def plot_loss(model: MLPClassifier, show: bool = False):
    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / "model_loss.png"
    fig, ax = plt.subplots()
    ax.plot(model.loss_curve_)
    ax.set(title="Model loss", xlabel="Iteration", ylabel="Loss")
    fig.savefig(out, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=9, help="seed for split and model")
    parser.add_argument("--show", action="store_true", help="display the loss plot")
    parser.add_argument("--no-plot", action="store_true", help="skip saving the loss plot")
    args = parser.parse_args(argv)

    feats = load_features()
    # Utterances from the training dialogues are split randomly into train / held-out sets.
    X, y = audio_dataset(feats, feats.train_ids)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    print(f"Train / test samples: {len(X_train)} / {len(X_test)}")
    print(f"Features per sample:  {X_train.shape[1]}")

    baseline = DummyClassifier(strategy="stratified", random_state=3).fit(X_train, y_train)
    print(f"Baseline (stratified dummy) accuracy: {baseline.score(X_test, y_test):.2%}")

    model = build_model(args.seed).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"MLP accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
    print(classification_report(y_test, y_pred, digits=3))

    print(f"Best loss: {model.best_loss_:.2f} | iterations: {model.n_iter_} | "
          f"layers: {model.n_layers_} | output activation: {model.out_activation_}")

    if not args.no_plot:
        print(f"Loss curve saved to {plot_loss(model, show=args.show)}")


if __name__ == "__main__":
    main()
