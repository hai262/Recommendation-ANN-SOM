import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from minisom import MiniSom
from collections import Counter
import io

# Page config and sidebar background CSS
st.set_page_config(page_title="ANN & SOM Recommendation", layout='wide')
sidebar_bg = """
<style>
[data-testid="stSidebar"] > div:first-child {
  background-image: url('https://miro.medium.com/v2/resize:fit:1400/1*cJEECUHB3fbJx-hc_70Zmg.jpeg');
  background-size: cover;
  background-position: center;
}
/* Sidebar text color */
[data-testid="stSidebar"] * {
  color: darkblue !important;
}
</style>
"""
st.markdown(sidebar_bg, unsafe_allow_html=True)

# Sidebar navigation and data upload
st.sidebar.title("Navigation & Data")
section = st.sidebar.radio("Go to", [
    "Home",
    "Introduction",
    "Data Overview",
    "Autoencoder",
    "Recommendations",
    "SOM Clustering",
    "Comparison"
])
file_upload = st.sidebar.file_uploader("Upload MovieLens data (u.data)", type=['csv','txt'])

# Helper functions
def load_movielens(path=None):
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    return pd.read_csv(path, sep='\t', names=cols) if path else pd.read_csv('u.data', sep='\t', names=cols)

def preprocess(df):
    user_item = df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    return user_item, user_item / 5.0

def train_autoencoder(X_train, latent_dim, epochs, batch_size):
    num_movies = X_train.shape[1]
    model = Sequential([
        Input(shape=(num_movies,)),
        Dense(512, activation='relu'),
        Dense(latent_dim, activation='relu'),
        Dense(512, activation='relu'),
        Dense(num_movies, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
    return model, history

def train_som(data, grid_rows, grid_cols, sigma, lr, iterations):
    som = MiniSom(grid_rows, grid_cols, data.shape[1], sigma=sigma, learning_rate=lr)
    som.random_weights_init(data)
    som.train(data, iterations, verbose=False)
    return som

# Home section
if section == "Home":
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] {
      background-image: url('https://cdn.ceps.eu/wp-content/uploads/2024/07/vecteezy_ai-generated-ai-circuit-board-technology-background_37348385-scaled.jpg');
      background-size: cover;
      background-position: center;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown("<h1 style='color:yellow; text-align:center; margin-top:200px;'>Recommendation System: ANN Autoencoder & SOM Clustering</h1>", unsafe_allow_html=True)
    st.markdown("---")

# Introduction section
elif section == "Introduction":
    st.title(":blue[Technical Overview & Motivation]")
    st.markdown(
        """
        Recommendation systems filter and suggest items (movies, products, music) by predicting user preferences from historical data. Widely used in e-commerce (Amazon), streaming (Netflix, Spotify), and social media (YouTube).
        
        **Autoencoder (ANN)**  
        - Learns a compressed latent representation of user–item interactions via an encoder–decoder architecture.  
        - Handles sparse matrices by effectively reconstructing missing ratings.  
        - Proven in winning the Netflix Prize with deep learning enhancements.
        
        **Self-Organizing Map (SOM)**  
        - Unsupervised, competitive neural network that maps high-dimensional data to a 2D grid.  
        - Preserves topological relations: similar users cluster together.  
        - Visualizes latent structures and identifies user segments or atypical behavior.
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

# Other sections
else:
    with st.spinner('Loading data...'):
        df = load_movielens(file_upload)
        user_item, normalized = preprocess(df)
        X = normalized.values
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    if section == "Data Overview":
        st.subheader(":blue[Data Overview]")
        st.write(df.head())
        st.write(f"Users: {user_item.shape[0]}, Movies: {user_item.shape[1]}")
        fig, ax = plt.subplots(figsize=(4,3))
        ax.hist(df['rating'], bins=np.arange(0.5,6.5,1), edgecolor='black')
        ax.set(title='Rating Distribution', xlabel='Rating', ylabel='Count')
        st.pyplot(fig)
        mr, md, sk = df['rating'].mean(), df['rating'].median(), df['rating'].skew()
        st.markdown(f"**Mean:** {mr:.2f}, **Median:** {md:.2f}, **Skew:** {sk:.2f}")

    elif section == "Autoencoder":
        st.subheader(":blue[Autoencoder Training]")
        c1, c2 = st.columns([1,4])
        with c1:
            latent_dim = st.slider("Latent dim", 64, 512, 256, 64)
            epochs = st.slider("Epochs", 10, 100, 50, 10)
            batch_size = st.selectbox("Batch size", [16,32,64], index=0)
        with c2:
            if st.button("Train Autoencoder"):
                with st.spinner('Training...'):
                    model, history = train_autoencoder(X_train, latent_dim, epochs, batch_size)
                    st.session_state['autoencoder_model'] = model
                st.success("Autoencoder trained!")
                buf = io.StringIO()
                model.summary(print_fn=lambda x: buf.write(x + "\n"))
                st.text("Model Architecture and Parameters:")
                st.text(buf.getvalue())
                fig2, ax2 = plt.subplots(figsize=(4,3))
                ax2.plot(history.history['loss'], label='Train')
                ax2.plot(history.history['val_loss'], label='Val')
                ax2.set(title='Loss Curves', xlabel='Epoch', ylabel='MSE')
                ax2.legend(fontsize=6)
                st.pyplot(fig2)
                ft = history.history['loss'][-1]
                fv = history.history['val_loss'][-1]
                be = np.argmin(history.history['val_loss']) + 1
                bv = np.min(history.history['val_loss'])
                st.write(f"Final Train: {ft:.4f}, Final Val: {fv:.4f}")
                st.write(f"Best Val {bv:.4f} at epoch {be}")

    elif section == "Recommendations":
        st.subheader(":blue[Recommendations]")
        model = st.session_state.get('autoencoder_model')
        if not model:
            st.warning("Please train autoencoder first.")
        else:
            uid = st.selectbox("User ID", user_item.index.tolist())
            if st.button("Recommend Top-5"):
                preds = model.predict(normalized.loc[[uid]].values)[0]
                preds[user_item.loc[uid] > 0] = -1
                top5 = np.argsort(preds)[-5:][::-1]
                st.write(top5.tolist())

    elif section == "SOM Clustering":
        st.subheader(":blue[SOM Clustering]")
        c1, c2 = st.columns([1,4])
        with c1:
            grid_rows = st.selectbox("Rows", [5,10,15], index=1)
            grid_cols = st.selectbox("Cols", [5,10,15], index=1)
            sigma = st.slider("Sigma", 0.5, 2.0, 1.0, 0.1)
            lr = st.slider("LR", 0.1, 1.0, 0.5, 0.1)
            iters = st.slider("Iters", 50, 500, 100, 50)
        with c2:
            if st.button("Train & Plot SOM"):
                som = train_som(X, grid_rows, grid_cols, sigma, lr, iters)
                st.success("SOM trained!")
                fig3, ax3 = plt.subplots(figsize=(4,3))
                um = som.distance_map().T
                ax3.pcolor(um, cmap='viridis')
                ax3.set(title='U-Matrix')
                st.pyplot(fig3)
                st.write(
                    f"Distances: min {um.min():.4f}, max {um.max():.4f}, mean {um.mean():.4f}"
                )
                fig4, ax4 = plt.subplots(figsize=(4,3))
                markers = ['o','s','D','^','v']
                colors = ['r','g','b','c','m']
                for i, x in enumerate(X):
                    r, c = som.winner(x)
                    ax4.plot(
                        c+0.5, r+0.5, markers[i%5], mfc='None',
                        mec=colors[i%5], ms=6, mew=1
                    )
                ax4.set(xlim=(0,grid_cols), ylim=(0,grid_rows))
                ax4.invert_yaxis()
                ax4.set(title='BMU Nodes')
                st.pyplot(fig4)
                freq = Counter([som.winner(x) for x in X])
                st.write(
                    f"Unique BMUs: {len(freq)}, Max pop: {freq.most_common(1)[0][1]}"
                )

    elif section == "Comparison":
        st.subheader(":blue[Performance Comparison]")
        model = st.session_state.get('autoencoder_model')
        if not model:
            st.info("Train autoencoder first.")
        else:
            base = np.tile(X_train.mean(0), (len(X_test),1))
            b_mse = mean_squared_error(X_test, base)
            b_rmse = np.sqrt(b_mse)
            pred = model.predict(X_test)
            a_mse = mean_squared_error(X_test, pred)
            a_rmse = np.sqrt(a_mse)
            c1, c2 = st.columns([1,1])
            with c1:
                st.write(f"Baseline RMSE: {b_rmse:.4f}")
                st.write(f"ANN RMSE: {a_rmse:.4f}")
                st.write(
                    f"Improvement: {(b_rmse-a_rmse)/b_rmse*100:.2f}%"
                )
            with c2:
                st.bar_chart({'Baseline': b_rmse, 'ANN': a_rmse})
