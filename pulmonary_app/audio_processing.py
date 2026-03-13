import librosa
import numpy as np

def extract_features(audio_path):
    """
    Extracts MFCC features from an audio file.
    Returns the mean of 13 MFCC coefficients.
    """
    try:
        # Load audio with librosa
        # sr=None preserves original sampling rate
        y, sr = librosa.load(audio_path, sr=22050)
        
        # If the audio is completely silent
        if np.max(np.abs(y)) == 0:
            return None
        
        # Extract MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        return mfccs_mean
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None
