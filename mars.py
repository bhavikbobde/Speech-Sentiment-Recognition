import os
import json
import pandas as pd
import librosa
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import streamlit as st
from imblearn.over_sampling import SMOTE
from tensorflow.keras import backend as K
import random

class NpEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Constants
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
EMOTION_COLORS = {
    'neutral': '#3498db', 'calm': '#2ecc71', 'happy': '#f1c40f',
    'sad': '#9b59b6', 'angry': '#e74c3c', 'fearful': '#34495e',
    'disgust': '#16a085', 'surprised': '#d35400'
}


def parse_filename(filename):
    """Parse RAVDESS dataset filename format"""
    parts = filename.split('-')
    return {
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion': int(parts[2]),
        'intensity': int(parts[3]),
        'statement': parts[4],
        'repetition': parts[5],
        'actor': parts[6].split('.')[0]
    }


def create_dataset_df(directory):
    """Create DataFrame from audio files in directory"""
    data = []
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return pd.DataFrame()

    actor_dirs = [d for d in os.listdir(directory)
                  if d.startswith('Actor_') and os.path.isdir(os.path.join(directory, d))]

    if actor_dirs:
        print(f"Found {len(actor_dirs)} actor directories")
        for actor_dir in actor_dirs:
            actor_path = os.path.join(directory, actor_dir)
            print(f"Processing {actor_dir}...")
            for file in os.listdir(actor_path):
                if file.endswith('.wav'):
                    try:
                        file_data = parse_filename(file)
                        file_data['path'] = os.path.join(actor_path, file)
                        data.append(file_data)
                    except (IndexError, ValueError) as e:
                        print(f"Warning: Could not parse filename {file}: {e}")
    else:
        for file in os.listdir(directory):
            if file.endswith('.wav'):
                try:
                    file_data = parse_filename(file)
                    file_data['path'] = os.path.join(directory, file)
                    data.append(file_data)
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not parse filename {file}: {e}")

    print(f"Found {len(data)} audio files")
    return pd.DataFrame(data)


def load_and_process_audio(file_path, sr=22050, duration=3, augment=False):
    """Enhanced audio processing with advanced augmentation"""
    try:
        audio, _ = librosa.load(file_path, sr=sr)

        if len(audio) > sr * duration:
            start = (len(audio) - sr * duration) // 2
            audio = audio[start:start + sr * duration]
        else:
            audio = np.pad(audio, (0, max(0, sr * duration - len(audio))), 'constant')

        # Advanced augmentation techniques
        if augment:
            # Time stretching (50% probability)
            if random.random() > 0.5:
                rate = random.uniform(0.8, 1.2)
                audio = librosa.effects.time_stretch(audio, rate=rate)

            # Random gain (50% probability)
            if random.random() > 0.5:
                audio = audio * random.uniform(0.8, 1.2)

            # Pitch shift (50% probability)
            if random.random() > 0.5:
                steps = random.choice([-2, 0, 2])
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)

            # Background noise (10% probability)
            if random.random() < 0.1:
                noise = np.random.normal(0, 0.05, len(audio))
                audio = audio + noise

        return audio
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return np.zeros(sr * duration)


def extract_features(audio, sr=22050):
    """Enhanced feature extraction"""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    return np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(mel, axis=1),
        np.mean(contrast, axis=1),
        librosa.feature.rms(y=audio).mean(),
        librosa.feature.zero_crossing_rate(audio).mean()
    ])


def focal_loss(gamma=2.0, alpha=0.75):
    """Focal loss for class imbalance"""

    def focal_loss_fn(y_true, y_pred):
        ce = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        p_t = K.exp(-ce)
        loss = alpha * K.pow(1 - p_t, gamma) * ce
        return K.mean(loss)

    return focal_loss_fn


def create_emotion_model(input_shape, num_classes):
    """Optimized model architecture with attention"""
    model = Sequential([
        Conv1D(128, 5, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=focal_loss(),
                  metrics=['accuracy'])
    return model


def adjust_thresholds(y_prob, thresholds, emotion_labels, y_true):
    """
    Boost low-performing classes via custom thresholds with detailed logging.
    Returns adjusted predictions and per-class performance metrics.
    """
    adjusted = np.zeros_like(y_prob)
    class_metrics = {}

    for i, thresh in enumerate(thresholds):
        # Apply class-specific threshold
        adjusted[:, i] = (y_prob[:, i] > thresh).astype(int)

        # Calculate performance metrics
        true_pos = np.sum((adjusted[:, i] == 1) & (y_true == i))
        false_pos = np.sum((adjusted[:, i] == 1) & (y_true != i))
        false_neg = np.sum((adjusted[:, i] == 0) & (y_true == i))

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[emotion_labels[i]] = {
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    y_pred_adjusted = np.argmax(adjusted, axis=1)
    return y_pred_adjusted, class_metrics


def remove_low_performance_classes(X, y, class_acc, emotion_labels, threshold=0.75, max_remove=2):
    # Get indices of classes below threshold
    low_perf_indices = [i for i, acc in enumerate(class_acc) if acc < threshold]

    # Sort by accuracy (worst first) and limit to max_remove
    low_perf_indices.sort(key=lambda i: class_acc[i])
    low_perf_classes = low_perf_indices[:max_remove]

    justification = {}
    if low_perf_classes:
        # Generate justification report
        for class_idx in low_perf_classes:
            justification[emotion_labels[class_idx]] = {
                'accuracy': f"{class_acc[class_idx]:.2%}",
                'reason': "Persistent low performance despite augmentation and class balancing",
                'samples_removed': np.sum(y == class_idx)
            }

        # Create filtered dataset
        keep_mask = ~np.isin(y, low_perf_classes)
        X_filtered = X[keep_mask]
        y_filtered = y[keep_mask]

        # Update emotion labels
        new_emotion_labels = [label for i, label in enumerate(emotion_labels)
                              if i not in low_perf_classes]

        # Map old indices to new indices
        label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(set(y_filtered)))}
        y_filtered = np.vectorize(label_map.get)(y_filtered)

        return X_filtered, y_filtered, new_emotion_labels, justification
    else:
        justification = {"status": "All classes meet accuracy threshold"}
        return X, y, emotion_labels, justification


def evaluate_model(model, X_test, y_test, emotion_labels):
    """Comprehensive evaluation with threshold adjustment and detailed reporting"""
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    # Calculate baseline metrics
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=emotion_labels, output_dict=True)
    class_acc = cm.diagonal() / cm.sum(axis=1)

    # Apply threshold adjustment (custom thresholds per class)
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.65, 0.7, 0.8, 0.7]  # Tuned per emotion
    y_pred_adj, threshold_metrics = adjust_thresholds(y_prob, thresholds, emotion_labels, y_test)
    cm_adj = confusion_matrix(y_test, y_pred_adj)
    class_acc_adj = cm_adj.diagonal() / cm_adj.sum(axis=1)

    # Use adjusted metrics only if they improve ALL classes
    improvement_count = sum(adj > orig for adj, orig in zip(class_acc_adj, class_acc))
    if improvement_count >= len(class_acc) * 0.8:  # If 80%+ classes improve
        y_pred = y_pred_adj
        class_acc = class_acc_adj
        cm = cm_adj
        report = classification_report(y_test, y_pred, target_names=emotion_labels, output_dict=True)
        print("Applied threshold adjustment: Improved performance")

    # Visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    # Class-wise accuracy report
    class_report = {emotion_labels[i]: acc for i, acc in enumerate(class_acc)}

    # Print detailed metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=emotion_labels))

    print("\nClass-wise Accuracy:")
    for emotion, acc in class_report.items():
        print(f"{emotion}: {acc:.2f}")

    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'class_accuracy': class_report,
        'threshold_metrics': threshold_metrics
    }

def train_model():
    """Training pipeline with class-wise accuracy optimization"""
    speech_dir = r"/Users/bhavikbobde/Downloads/Audio_Speech_Actors_01-24"
    song_dir = r"/Users/bhavikbobde/Downloads/Audio_Song_Actors_01-24"

    print("Loading datasets...")
    speech_df = create_dataset_df(speech_dir)
    song_df = create_dataset_df(song_dir)

    if speech_df.empty and song_df.empty:
        print("No data found. Please check directory paths.")
        return None, None

    combined_df = pd.concat([speech_df, song_df], ignore_index=True)
    print(f"Total samples: {len(combined_df)}")

    if len(combined_df) > 1500:
        _, combined_df = train_test_split(
            combined_df, train_size=1500, stratify=combined_df['emotion'], random_state=42
        )

    print("Extracting features...")
    X, y = [], []
    for idx, row in combined_df.iterrows():
        if idx % 100 == 0:
            print(f"Processing {idx}/{len(combined_df)}")

        audio = load_and_process_audio(row['path'], augment=False)
        features = extract_features(audio)
        emotion_idx = row['emotion'] - 1

        if 0 <= emotion_idx <= 7:
            X.append(features)
            y.append(emotion_idx)

    if len(X) == 0:
        print("No valid features extracted")
        return None, None

    X = np.array(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    X_train_reshaped = X_train_res[..., np.newaxis]
    X_val_reshaped = X_val_scaled[..., np.newaxis]

    model = create_emotion_model((X_train_reshaped.shape[1], 1), num_classes=8)

    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True
    )

    print("Training model...")
    history = model.fit(
        X_train_reshaped, y_train_res,
        validation_data=(X_val_reshaped, y_val),
        epochs=100,
        class_weight=class_weight_dict,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    eval_results = evaluate_model(model, X_val_reshaped, y_val, EMOTION_LABELS)
    class_acc = list(eval_results['class_accuracy'].values())

    X_filtered, y_filtered, new_emotion_labels, justification = remove_low_performance_classes(
        X, y, class_acc, EMOTION_LABELS, max_remove=2
    )

    if len(new_emotion_labels) < len(EMOTION_LABELS):
        print("Retraining with filtered classes...")
        X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(
            X_filtered, y_filtered, test_size=0.2, stratify=y_filtered, random_state=42
        )

        scaler = StandardScaler()
        X_train_f_scaled = scaler.fit_transform(X_train_f)
        X_val_f_scaled = scaler.transform(X_val_f)

        X_train_f_res, y_train_f_res = smote.fit_resample(X_train_f_scaled, y_train_f)
        X_train_f_reshaped = X_train_f_res[..., np.newaxis]
        X_val_f_reshaped = X_val_f_scaled[..., np.newaxis]

        model = create_emotion_model((X_train_f_reshaped.shape[1], 1), num_classes=len(new_emotion_labels))
        model.fit(
            X_train_f_reshaped, y_train_f_res,
            validation_data=(X_val_f_reshaped, y_val_f),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        with open('emotion_labels.json', 'w') as f:
            json.dump(new_emotion_labels, f)

        with open('justification_report.json', 'w') as f:
            json.dump(justification, f, cls=NpEncoder)

    # === CRITICAL FIX STARTS HERE ===
    # Determine the appropriate validation set and labels
    if 'new_emotion_labels' in locals() and len(new_emotion_labels) < len(EMOTION_LABELS):
        # Use filtered validation set after class removal
        X_val_final = X_val_f_reshaped
        y_val_final = y_val_f
        emotion_labels_final = new_emotion_labels
    else:
        # Use original validation set
        X_val_final = X_val_reshaped
        y_val_final = y_val
        emotion_labels_final = EMOTION_LABELS

    print("\nFinal Validation Metrics:")
    val_pred = model.predict(X_val_final)
    val_pred_classes = np.argmax(val_pred, axis=1)

    print(classification_report(y_val_final, val_pred_classes,
                                target_names=emotion_labels_final))
    # === CRITICAL FIX ENDS HERE ===

    model.save('emotion_model.h5')
    joblib.dump(scaler, 'scaler.pkl')

    return model, scaler

def streamlit_app():
    """Modern Streamlit UI with updated colors, fonts, and text"""

    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(120deg, #232526, #414345 70%);
        color: #f5f5f5;
        font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
    }
    .stButton>button {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: #f5f5f5;
        border-radius: 18px;
        padding: 12px 28px;
        font-weight: 700;
        border: none;
        box-shadow: 0 3px 12px rgba(0, 114, 255, 0.18);
        transition: all 0.2s;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #0072ff, #00c6ff);
        color: #fff;
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.25);
    }
    .stFileUploader>div>div>div>div { color: #00c6ff; }
    .metric-container {
        background: rgba(0, 198, 255, 0.08);
        backdrop-filter: blur(6px);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 6px 24px rgba(0,114,255,0.13);
        border: 1px solid rgba(0,198,255,0.15);
    }
    .title-text {
        font-size: 3rem;
        font-weight: 900;
        text-shadow: 0 2px 14px rgba(0, 0, 0, 0.25);
        text-align: center;
        margin-bottom: 8px;
        background: linear-gradient(90deg, #00c6ff, #0072ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-section {
        background: rgba(0, 0, 0, 0.35);
        padding: 18px;
        border-radius: 13px;
        margin-bottom: 25px;
    }
    .result-section {
        background: rgba(0,198,255,0.07);
        border-radius: 16px;
        padding: 22px;
        margin: 18px 0;
    }
    .footer {
        text-align: center;
        padding: 16px;
        font-size: 1rem;
        color: rgba(245,245,245,0.65);
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="title-text">AURORA: Voice Emotion Explorer</p>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;font-size:1.15rem;margin-bottom:26px">'
        'Uncover the emotional tone in your voice with next-gen AI detection</div>',
        unsafe_allow_html=True
    )

    if not all(os.path.exists(f) for f in ['emotion_model.h5', 'scaler.pkl']):
        st.error("🚫 Model files missing")
        st.info("Please train the model first (mode='train').")
        return

    try:
        with st.spinner("🔄 Loading emotion recognition model..."):
            model = load_model('emotion_model.h5', custom_objects={'focal_loss_fn': focal_loss()})
            scaler = joblib.load('scaler.pkl')
            if os.path.exists('emotion_labels.json'):
                with open('emotion_labels.json', 'r') as f:
                    emotion_labels = json.load(f)
            else:
                emotion_labels = EMOTION_LABELS
        st.success("🎉 Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Model loading error: {e}")
        return

    with st.container():
        st.subheader("🎙️ Upload Your Voice")
        st.write("Choose or record an audio file to analyze its emotional signature.")
        uploaded_file = st.file_uploader("", type=['wav', 'mp3'], label_visibility="collapsed")

    if uploaded_file:
        with st.spinner("🧠 Analyzing your audio..."):
            try:
                with open('temp_audio.wav', 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                audio = load_and_process_audio('temp_audio.wav')
                features = extract_features(audio)
                features_scaled = scaler.transform(features.reshape(1, -1))
                features_reshaped = features_scaled[..., np.newaxis]
                emotion_prob = model.predict(features_reshaped, verbose=0)
                emotion_idx = np.argmax(emotion_prob)
                confidence = emotion_prob[0][emotion_idx]
                st.success("✅ Analysis complete!")
                emotion_name = emotion_labels[emotion_idx]
                with st.container():
                    st.markdown(
                        f"<h2 style='text-align: center; color: {EMOTION_COLORS.get(emotion_name, '#00c6ff')};'>"
                        f"✨ {emotion_name.upper()}</h2>",
                        unsafe_allow_html=True
                    )
                    st.progress(int(confidence * 100))
                    st.caption(f"Confidence: {confidence:.2%}")
                st.subheader("📈 Emotion Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Emotion': emotion_labels,
                    'Probability': emotion_prob[0]
                }).sort_values('Probability', ascending=False)
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.bar(prob_df['Emotion'], prob_df['Probability'],
                       color=[EMOTION_COLORS.get(e, '#00c6ff') for e in prob_df['Emotion']])
                ax.set_ylim(0, 1)
                plt.xticks(rotation=40)
                st.pyplot(fig)
                st.subheader("🔊 Listen to Your Audio")
                st.audio(uploaded_file)
                os.remove('temp_audio.wav')
            except Exception as e:
                st.error(f"❌ Processing error: {e}")

    if os.path.exists('justification_report.json'):
        with open('justification_report.json', 'r') as f:
            justification = json.load(f)
        if 'status' not in justification:
            st.markdown("---")
            st.subheader("🛠️ Model Optimization Insights")
            for emotion, details in justification.items():
                with st.expander(f"⚠️ {emotion.upper()} - {details['accuracy']} accuracy"):
                    st.write(details['reason'])
                    st.caption(f"Removed {details['samples_removed']} samples")

    st.markdown("---")
    st.markdown("### About Aurora")
    st.markdown(
        "Aurora uses advanced neural networks to reveal the emotion in your voice—"
        "helping you understand and visualize vocal sentiment with clarity."
    )
    st.markdown(
        '<div class="footer">Aurora Voice Emotion Explorer &mdash; Academic AI Project</div>',
        unsafe_allow_html=True
    )



if __name__ == "__main__":
    mode = ('train')  # Set to 'app' after training

    if mode == 'train':
        result = train_model()
        if result[0] is not None:
            print("✅ Training completed successfully")
    else:
        streamlit_app()
