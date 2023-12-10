import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import librosa

# Load the IEMOCAP dataset
data_path = 'IEMOCAP_features.pkl'

with open(data_path, 'rb') as file:
    (
        videoIDs, videoSpeakers, videoLabels, videoText,
        videoAudio, videoVisual, videoSentence, trainVid,
        testVid
    ) = pickle.load(file, encoding='latin1')

# Define the emotions mapping based on your dataset
emotions = {
    '0': 'happy',
    '1': 'sad',
    '2': 'neutral',
    '3': 'angry',
    '4': 'excited',
}

# Define the emotions to observe
observed_emotions = ['sad', 'happy', 'neutral', 'angry', 'excited']

"""# Extract features (mfcc, chroma, mel) from audio data
def extract_feature(audio_data, mfcc=True, chroma=True, mel=True):
    feature_vector = []

    for audio_segment in audio_data:
        audio_segment = np.array(audio_segment)

        # Extract features for each audio segment
        features = []

        # Extract MFCC if requested
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=audio_segment, sr=16000, n_mfcc=40).T, axis=0)
            features.extend(mfccs)

        # Extract chroma if requested
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(y=audio_segment, sr=16000).T, axis=0)
            features.extend(chroma)

        # Extract mel spectrogram if requested
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=audio_segment, sr=16000).T, axis=0)
            features.extend(mel)

        # Append the features of the current segment to the overall feature vector
        feature_vector.append(features)

    return feature_vector


features = extract_feature(videoAudio, mfcc=True, chroma=True, mel=True)"""

# Load the data and extract features
# Load the data and extract features
def load_data(trainVid, test_size=0.2):
    X, y = [], []
    for vid in videoIDs:
        if vid not in trainVid:
            continue
        for label, audio in zip(videoLabels[vid], videoAudio[vid]):
            label_str = str(label)
            if label_str == '5':
                # Skip data points with label '5'
                continue
            
            if label_str not in emotions:
                # Handle missing labels (skip or handle differently)
                continue
            
            emotion = emotions[label_str]
            if emotion not in observed_emotions:
                continue
            X.append(audio)
            y.append(emotion)
    return train_test_split(np.array(X), y, test_size=test_size, random_state=9)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = load_data(trainVid, test_size=0.25)

# Get the shape of the training and testing datasets
print((X_train.shape[0], X_test.shape[0]))

# Get the number of features extracted
print(f'Features extracted: {X_train.shape[1]}')

# Establish a baseline with Dummy Classifier
dummy_model = DummyClassifier(strategy='stratified', random_state=3)
dummy_model.fit(X_train, y_train)
dummy_model.predict(X_test)
print(f'Accuracy: {round(dummy_model.score(X_test, y_test), 2)*100}%')

# Initialize the Multi-Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(X_train, y_train)
# Predict for the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))

# Plot the loss curve
plt.plot(model.loss_curve_)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.savefig('visualizations/model_loss.png')
plt.show()

# Inspect other attributes to get a sense of the model structure
print(f'Best loss: {round(model.best_loss_, 2)}')
print(f'Number of iterations: {model.n_iter_}')
print(f'Number of layers: {model.n_layers_}')
print(f'Output activation function: {model.out_activation_}')
