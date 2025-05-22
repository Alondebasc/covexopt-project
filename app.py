import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="ConvexOpt | Fitness Intelligence",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design futuriste
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, #ffffff, #e0e0e0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        font-weight: 300;
        letter-spacing: 2px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, #00f5ff, #0080ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .prediction-zone {
        background: linear-gradient(135deg, #134e5e 0%, #71b280 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
        border: none;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .plot-container {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Fonction de chargement des données avec cache
@st.cache_data
def load_data():
    """Charge et prétraite les données"""
    try:
        df = pd.read_csv('cleaned_dataset_1.csv')
        
        # Mapping des colonnes pour l'interprétation
        feature_mapping = {
            'Age': 'Âge',
            'Gender': 'Genre', 
            'Weight (kg)': 'Poids (kg)',
            'Height (m)': 'Taille (m)',
            'Max_BPM': 'Fréquence cardiaque max',
            'Avg_BPM': 'Fréquence cardiaque moyenne',
            'Resting_BPM': 'Fréquence cardiaque repos',
            'Session_Duration (hours)': 'Durée session (heures)',
            'Calories_Burned': 'Calories brûlées',
            'Workout_Type': 'Type d\'entraînement',
            'Fat_Percentage': 'Pourcentage de graisse',
            'Water_Intake (liters)': 'Consommation d\'eau (litres)',
            'Workout_Frequency (days/week)': 'Fréquence d\'entraînement (jours/semaine)',
            'Experience_Level': 'Niveau d\'expérience',
            'BMI': 'IMC'
        }
        
        return df, feature_mapping
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None, None

# Fonction pour créer des graphiques personnalisés
def create_custom_plotly_figure(fig, title):
    """Applique un style personnalisé aux figures Plotly"""
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'font': {'family': 'Orbitron', 'size': 20, 'color': '#2c3e50'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'},
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255,255,255,0.1)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        )
    )
    return fig

# Interface principale
def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">CONVEXOPT</h1>
        <p class="subtitle">Intelligence Artificielle & Optimisation Fitness</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement des données
    df, feature_mapping = load_data()
    
    if df is None:
        st.error("🚨 Impossible de charger les données. Vérifiez que le fichier 'cleaned_dataset_1.csv' est présent.")
        return
    
    # Sidebar avec navigation
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2 style="color: white; font-family: 'Orbitron';">🚀 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Sélectionnez une section",
        ["🏠 Vue d'ensemble", "📊 Analyse exploratoire", "🔬 Visualisations avancées", "🎯 Prédiction", "⚙️ Optimisation", "📈 Monitoring"]
    )
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Échantillons</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df.columns)}</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        unique_workouts = df['Workout_Type'].nunique() if 'Workout_Type' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{unique_workouts}</div>
            <div class="metric-label">Types d'entraînement</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_calories = df['Calories_Burned'].mean() if 'Calories_Burned' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_calories:.0f}</div>
            <div class="metric-label">Calories moy.</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation des pages
    if page == "🏠 Vue d'ensemble":
        show_overview(df, feature_mapping)
    elif page == "📊 Analyse exploratoire":
        show_exploratory_analysis(df, feature_mapping)
    elif page == "🔬 Visualisations avancées":
        show_advanced_visualizations(df, feature_mapping)
    elif page == "🎯 Prédiction":
        show_prediction_interface(df, feature_mapping)
    elif page == "⚙️ Optimisation":
        show_optimization_section(df, feature_mapping)
    elif page == "📈 Monitoring":
        show_monitoring_section(df, feature_mapping)

def show_overview(df, feature_mapping):
    """Affiche la vue d'ensemble du dataset"""
    st.markdown('<h2 class="section-header">🏠 Vue d\'ensemble du Dataset</h2>', unsafe_allow_html=True)
    
    # Onglets pour organiser l'information
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Informations", "🔍 Statistiques", "📈 Distribution", "🔗 Corrélations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-box">
                <h3>📊 Structure des données</h3>
                <p>• Dataset standardisé et normalisé</p>
                <p>• Variables continues et catégorielles</p>
                <p>• Données fitness & biométriques</p>
                <p>• Prêt pour l'optimisation ML</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Informations détaillées
            st.subheader("🔍 Détails techniques")
            info_df = pd.DataFrame({
                'Métrique': ['Lignes', 'Colonnes', 'Valeurs manquantes', 'Doublons', 'Taille mémoire'],
                'Valeur': [
                    f"{len(df):,}",
                    f"{len(df.columns)}",
                    f"{df.isnull().sum().sum()}",
                    f"{df.duplicated().sum()}",
                    f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                ]
            })
            st.dataframe(info_df, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="feature-box">
                <h3>🎯 Objectifs du projet</h3>
                <p>• Optimisation convexe avancée</p>
                <p>• Comparaison d'algorithmes (SGD, Adam...)</p>
                <p>• Interprétabilité des modèles</p>
                <p>• Interface interactive temps réel</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Preview des données
            st.subheader("👀 Aperçu des données (5 premières lignes)")
            st.dataframe(df.head(), use_container_width=True)
    
    with tab2:
        # Statistiques descriptives
        st.subheader("📈 Statistiques descriptives")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_df = df[numeric_cols].describe().round(3)
        st.dataframe(stats_df, use_container_width=True)
        
        # Distribution des types de données
        col1, col2 = st.columns(2)
        with col1:
            dtype_counts = df.dtypes.value_counts()
            fig = px.pie(values=dtype_counts.values, names=[str(d) for d in dtype_counts.index], 
                        title="Distribution des types de données")

            fig = create_custom_plotly_figure(fig, "Distribution des types de données")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Visualisation des valeurs manquantes
            missing_data = df.isnull().sum().sort_values(ascending=False)
            if missing_data.sum() > 0:
                fig = px.bar(x=missing_data.index, y=missing_data.values, 
                           title="Valeurs manquantes par variable")
                fig = create_custom_plotly_figure(fig, "Valeurs manquantes par variable")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ Aucune valeur manquante détectée!")
    
    with tab3:
        # Distribution des variables principales
        st.subheader("📊 Distributions des variables clés")
        
        # Sélection de variables à visualiser
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_vars = st.multiselect(
            "Sélectionnez les variables à analyser:",
            numeric_cols,
            default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
        )
        
        if selected_vars:
            # Créer des histogrammes
            n_cols = 2
            n_rows = (len(selected_vars) + 1) // 2
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=selected_vars,
                specs=[[{"secondary_y": False}] * n_cols for _ in range(n_rows)]
            )
            
            for i, var in enumerate(selected_vars):
                row = (i // n_cols) + 1
                col = (i % n_cols) + 1
                
                fig.add_trace(
                    go.Histogram(x=df[var], name=var, showlegend=False),
                    row=row, col=col
                )
            
            fig.update_layout(height=300*n_rows, showlegend=False)
            fig = create_custom_plotly_figure(fig, "Distributions des variables sélectionnées")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Matrice de corrélation
        st.subheader("🔗 Analyse des corrélations")
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = df.corr(numeric_only=True)

            # ➕ Conversion explicite
            corr_matrix = corr_matrix.astype(float)

            # ➕ Création de la figure corrigée
            fig = px.imshow(
                corr_matrix.to_numpy(),
                x=corr_matrix.columns.astype(str),
                y=corr_matrix.index.astype(str),
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Matrice de corrélation des variables numériques"
            )


            fig = create_custom_plotly_figure(fig, "Matrice de corrélation")
            st.plotly_chart(fig, use_container_width=True)

            
            # Top corrélations
            st.subheader("🔝 Corrélations les plus fortes")
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Corrélation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.reindex(corr_df['Corrélation'].abs().sort_values(ascending=False).index)
            st.dataframe(corr_df.head(10), use_container_width=True)

def show_exploratory_analysis(df, feature_mapping):
    """Section d'analyse exploratoire approfondie"""
    st.markdown('<h2 class="section-header">📊 Analyse Exploratoire Avancée</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Variables cibles", "🔍 Segmentation", "📈 Tendances"])
    
    with tab1:
        # Analyse des variables potentiellement cibles
        st.subheader("🎯 Identification des variables cibles potentielles")
        
        target_candidates = ['Calories_Burned', 'BMI', 'Fat_Percentage', 'Avg_BPM']
        available_targets = [col for col in target_candidates if col in df.columns]
        
        if available_targets:
            selected_target = st.selectbox("Sélectionnez une variable cible:", available_targets)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution de la variable cible
                fig = px.histogram(df, x=selected_target, nbins=30, 
                                 title=f"Distribution de {selected_target}")
                fig = create_custom_plotly_figure(fig, f"Distribution de {selected_target}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot par catégorie si disponible
                if 'Experience_Level' in df.columns:
                    fig = px.box(df, y=selected_target, x='Experience_Level',
                               title=f"{selected_target} par niveau d'expérience")
                    fig = create_custom_plotly_figure(fig, f"{selected_target} par niveau d'expérience")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques de la variable cible
            target_stats = df[selected_target].describe()
            st.subheader(f"📊 Statistiques pour {selected_target}")
            
            metric_cols = st.columns(5)
            metrics = ['mean', 'std', 'min', 'max', '50%']
            labels = ['Moyenne', 'Écart-type', 'Minimum', 'Maximum', 'Médiane']
            
            for i, (metric, label) in enumerate(zip(metrics, labels)):
                with metric_cols[i]:
                    st.metric(label, f"{target_stats[metric]:.2f}")
    
    with tab2:
        # Analyse de segmentation
        st.subheader("🔍 Segmentation des données")
        
        # Sélection des variables de segmentation
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if categorical_cols:
            segment_var = st.selectbox("Variable de segmentation:", categorical_cols)
            metric_var = st.selectbox("Métrique à analyser:", numeric_cols)
            
            # Analyse par segment
            segment_analysis = df.groupby(segment_var)[metric_var].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
            st.dataframe(segment_analysis, use_container_width=True)
            
            # Visualisation
            fig = px.violin(df, x=segment_var, y=metric_var,
                          title=f"Distribution de {metric_var} par {segment_var}")
            fig = create_custom_plotly_figure(fig, f"Distribution de {metric_var} par {segment_var}")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Analyse des tendances
        st.subheader("📈 Analyse des tendances et patterns")
        
        # Scatter plot matrix pour explorer les relations
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.multiselect(
            "Sélectionnez les variables pour l'analyse croisée:",
            numeric_cols,
            default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
        )
        
        if len(selected_features) >= 2:
            # Créer un scatter plot matrix
            fig = px.scatter_matrix(df[selected_features])
            fig = create_custom_plotly_figure(fig, "Matrice de nuages de points")
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse PCA si assez de variables
            if len(selected_features) >= 3:
                st.subheader("🔬 Analyse en Composantes Principales (PCA)")
                
                # Standardisation des données
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[selected_features].fillna(df[selected_features].mean()))
                
                # PCA
                pca = PCA()
                pca_result = pca.fit_transform(scaled_data)
                
                # Variance expliquée
                explained_var = pca.explained_variance_ratio_
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Graphique de la variance expliquée
                    fig = px.bar(x=range(1, len(explained_var)+1), 
                               y=explained_var,
                               title="Variance expliquée par composante")
                    fig = create_custom_plotly_figure(fig, "Variance expliquée par composante")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Projection sur les 2 premières composantes
                    pca_df = pd.DataFrame(pca_result[:, :2], columns=['PC1', 'PC2'])
                    fig = px.scatter(pca_df, x='PC1', y='PC2',
                                   title="Projection PCA (2 premières composantes)")
                    fig = create_custom_plotly_figure(fig, "Projection PCA")
                    st.plotly_chart(fig, use_container_width=True)

def show_advanced_visualizations(df, feature_mapping):
    """Visualisations avancées et interactives"""
    st.markdown('<h2 class="section-header">🔬 Visualisations Avancées</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🌐 3D & Interactif", "🔥 Heatmaps", "📊 Comparatifs"])
    
    with tab1:
        st.subheader("🌐 Visualisations 3D et interactives")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 3:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("Axe X:", numeric_cols, key="3d_x")
            with col2:
                y_var = st.selectbox("Axe Y:", numeric_cols, index=1, key="3d_y")
            with col3:
                z_var = st.selectbox("Axe Z:", numeric_cols, index=2, key="3d_z")
            
            # Graphique 3D scatter
            fig = px.scatter_3d(df, x=x_var, y=y_var, z=z_var,
                              title=f"Visualisation 3D: {x_var} vs {y_var} vs {z_var}")
            
            fig.update_layout(
                scene=dict(
                    bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
                    yaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
                    zaxis=dict(backgroundcolor='rgba(0,0,0,0)')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # t-SNE si le dataset n'est pas trop grand
            if len(df) <= 5000:
                st.subheader("🔬 Visualisation t-SNE")
                
                # Préparation des données pour t-SNE
                numeric_data = df[numeric_cols].fillna(df[numeric_cols].mean())
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                
                # t-SNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)//4))
                tsne_result = tsne.fit_transform(scaled_data[:1000])  # Limite pour la performance
                
                tsne_df = pd.DataFrame(tsne_result, columns=['t-SNE 1', 't-SNE 2'])
                fig = px.scatter(tsne_df, x='t-SNE 1', y='t-SNE 2',
                               title="Projection t-SNE des données")
                fig = create_custom_plotly_figure(fig, "Projection t-SNE")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔥 Heatmaps avancées")
        
        # Heatmap de corrélation avec clustering
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 2:
            corr_matrix = numeric_df.corr()
            
            # Dendrogramme et clustering
            fig = px.imshow(corr_matrix, 
                          text_auto=True,
                          color_continuous_scale='RdBu_r',
                          title="Matrice de corrélation avec clustering")
            fig = create_custom_plotly_figure(fig, "Matrice de corrélation avec clustering")
            st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap des statistiques par groupe
            if 'Experience_Level' in df.columns:
                st.subheader("📊 Heatmap des moyennes par groupe")
                
                group_stats = df.groupby('Experience_Level')[numeric_df.columns].mean()
                
                fig = px.imshow(group_stats.T,
                              text_auto=True,
                              aspect="auto",
                              title="Moyennes des variables par niveau d'expérience")
                fig = create_custom_plotly_figure(fig, "Moyennes par niveau d'expérience")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Analyses comparatives")
        
        # Comparaison de distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            var1 = st.selectbox("Variable 1:", numeric_cols, key="comp_var1")
            var2 = st.selectbox("Variable 2:", numeric_cols, index=1, key="comp_var2")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distributions superposées
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df[var1], name=var1, opacity=0.7))
                fig.add_trace(go.Histogram(x=df[var2], name=var2, opacity=0.7))
                fig.update_layout(barmode='overlay', title=f"Comparaison des distributions: {var1} vs {var2}")
                fig = create_custom_plotly_figure(fig, f"Distributions: {var1} vs {var2}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Scatter plot avec régression
                fig = px.scatter(df, x=var1, y=var2, trendline="ols",
                               title=f"Relation entre {var1} et {var2}")
                fig = create_custom_plotly_figure(fig, f"Relation: {var1} vs {var2}")
                st.plotly_chart(fig, use_container_width=True)
            
            # Comparaison statistique
            st.subheader("📈 Comparaison statistique")
            
            comparison_stats = pd.DataFrame({
                'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Min', 'Max', 'Asymétrie', 'Aplatissement'],
                var1: [
                    df[var1].mean(),
                    df[var1].median(),
                    df[var1].std(),
                    df[var1].min(),
                    df[var1].max(),
                    df[var1].skew(),
                    df[var1].kurtosis()
                ],
                var2: [
                    df[var2].mean(),
                    df[var2].median(),
                    df[var2].std(),
                    df[var2].min(),
                    df[var2].max(),
                    df[var2].skew(),
                    df[var2].kurtosis()
                ]
            }).round(3)
            
            st.dataframe(comparison_stats, use_container_width=True)

def show_prediction_interface(df, feature_mapping):
    """Interface de prédiction interactive"""
    st.markdown('<h2 class="section-header">🎯 Interface de Prédiction</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="prediction-zone">
        <h3>🚀 Zone de Prédiction Interactive</h3>
        <p>Cette section sera intégrée avec les modèles développés par l'équipe.</p>
        <p>Fonctionnalités prévues:</p>
        <ul>
            <li>✅ Sélection de profils utilisateurs</li>
            <li>✅ Prédictions en temps réel</li>
            <li>✅ Visualisation des résultats</li>
            <li>✅ Analyse de sensibilité</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["👤 Profil utilisateur", "🔮 Prédiction", "📊 Analyse"])
    
    with tab1:
        st.subheader("👤 Configuration du profil utilisateur")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Paramètres utilisateur
            st.markdown("### 📝 Informations personnelles")
            
            age = st.slider("Âge", 18, 80, 30)
            gender = st.selectbox("Genre", ["Homme", "Femme"])
            weight = st.slider("Poids (kg)", 40, 150, 70)
            height = st.slider("Taille (cm)", 140, 220, 170)
            
            # Calcul automatique de l'IMC
            bmi = weight / ((height/100) ** 2)
            st.metric("IMC calculé", f"{bmi:.1f}")
            
        with col2:
            st.markdown("### 🏋️ Paramètres d'entraînement")
            
            experience = st.selectbox("Niveau d'expérience", 
                                    ["Débutant", "Intermédiaire", "Avancé", "Expert"])
            workout_type = st.selectbox("Type d'entraînement préféré", 
                                      ["Cardio", "Musculation", "HIIT", "Yoga", "Crossfit"])
            frequency = st.slider("Fréquence (jours/semaine)", 1, 7, 3)
            duration = st.slider("Durée moyenne (heures)", 0.5, 3.0, 1.0, 0.5)
            
        # Affichage du profil créé
        st.subheader("📋 Profil utilisateur créé")
        
        user_profile = {
            "Âge": age,
            "Genre": gender,
            "Poids": f"{weight} kg",
            "Taille": f"{height} cm",
            "IMC": f"{bmi:.1f}",
            "Expérience": experience,
            "Type d'entraînement": workout_type,
            "Fréquence": f"{frequency} jours/semaine",
            "Durée": f"{duration} heures"
        }
        
        profile_df = pd.DataFrame(list(user_profile.items()), columns=['Paramètre', 'Valeur'])
        st.dataframe(profile_df, use_container_width=True)
    
    with tab2:
        st.subheader("🔮 Simulation de prédiction")
        
        # Simulation de prédiction (à remplacer par le vrai modèle)
        if st.button("🚀 Lancer la prédiction", type="primary"):
            
            # Simulation de calculs
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import time
            
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text('🔄 Préparation des données...')
                elif i < 60:
                    status_text.text('🧠 Calcul des prédictions...')
                elif i < 90:
                    status_text.text('📊 Analyse des résultats...')
                else:
                    status_text.text('✅ Prédiction terminée!')
                time.sleep(0.02)
            
            # Résultats simulés
            col1, col2, col3 = st.columns(3)
            
            with col1:
                predicted_calories = np.random.randint(200, 800)
                st.metric("Calories prédites", f"{predicted_calories} kcal", 
                         delta=f"+{np.random.randint(10, 50)} vs moyenne")
            
            with col2:
                predicted_performance = np.random.uniform(0.7, 0.95)
                st.metric("Score de performance", f"{predicted_performance:.2f}", 
                         delta=f"+{np.random.uniform(0.01, 0.1):.2f}")
            
            with col3:
                predicted_risk = np.random.uniform(0.1, 0.3)
                st.metric("Risque de blessure", f"{predicted_risk:.2f}", 
                         delta=f"-{np.random.uniform(0.01, 0.05):.2f}")
            
            # Graphique de prédiction
            st.subheader("📈 Évolution prédite")
            
            days = np.arange(1, 31)
            predicted_progress = np.cumsum(np.random.normal(2, 1, 30)) + 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=days, y=predicted_progress, 
                                   mode='lines+markers', name='Progression prédite',
                                   line=dict(color='#667eea', width=3)))
            
            fig.update_layout(
                title='Progression fitness prédite (30 jours)',
                xaxis_title='Jours',
                yaxis_title='Score de fitness',
                hovermode='x unified'
            )
            
            fig = create_custom_plotly_figure(fig, 'Progression fitness prédite')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Analyse de sensibilité")
        
        st.markdown("""
        <div class="feature-box">
            <h4>🔍 Analyse d'impact des variables</h4>
            <p>Cette section analysera l'impact de chaque variable sur les prédictions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulation d'analyse de sensibilité
        variables = ['Âge', 'Poids', 'Fréquence', 'Durée', 'Expérience']
        importance = np.random.dirichlet(np.ones(len(variables))) * 100
        
        # Graphique d'importance des variables
        fig = px.bar(x=variables, y=importance, 
                    title="Importance des variables sur la prédiction")
        fig = create_custom_plotly_figure(fig, "Importance des variables")
        st.plotly_chart(fig, use_container_width=True)
        
        # Analyse SHAP simulée
        st.subheader("🎯 Analyse SHAP (simulée)")
        
        shap_values = np.random.normal(0, 1, len(variables))
        colors = ['red' if x < 0 else 'green' for x in shap_values]
        
        fig = go.Figure(go.Bar(
            x=shap_values,
            y=variables,
            orientation='h',
            marker=dict(color=colors)
        ))
        
        fig.update_layout(
            title='Valeurs SHAP - Impact sur la prédiction',
            xaxis_title='Impact SHAP',
            yaxis_title='Variables'
        )
        
        fig = create_custom_plotly_figure(fig, 'Valeurs SHAP')
        st.plotly_chart(fig, use_container_width=True)

def show_optimization_section(df, feature_mapping):
    """Section dédiée à l'optimisation et aux algorithmes"""
    st.markdown('<h2 class="section-header">⚙️ Optimisation & Algorithmes</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📐 Gradient Descent", "🚀 Optimizers", "📊 Convergence", "🔬 Expérimentations"])
    
    with tab1:
        st.subheader("📐 Visualisation du Gradient Descent")
        
        st.markdown("""
        <div class="feature-box">
            <h4>🎯 Simulation interactive du Gradient Descent</h4>
            <p>Cette section intégrera les implémentations manuelles de l'équipe.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Paramètres de simulation
        col1, col2 = st.columns(2)
        
        with col1:
            learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
            n_iterations = st.slider("Nombre d'itérations", 10, 100, 50)
            
        with col2:
            optimizer_type = st.selectbox("Type d'optimiseur", 
                                        ["SGD", "Batch GD", "Mini-batch GD", "Adam", "RMSprop"])
            batch_size = st.slider("Taille du batch", 1, 100, 32)
        
        # Simulation de la descente de gradient
        if st.button("🚀 Simuler la descente de gradient"):
            
            # Génération de données simulées pour la démonstration
            np.random.seed(42)
            x = np.linspace(-10, 10, 100)
            y = x**2 + 2*x + 1 + np.random.normal(0, 5, 100)  # Fonction quadratique avec bruit
            
            # Simulation de l'évolution des paramètres
            iterations = np.arange(n_iterations)
            loss_history = []
            
            # Simulation de la perte qui diminue
            initial_loss = 1000
            for i in iterations:
                if optimizer_type == "Adam":
                    # Adam converge plus rapidement
                    loss = initial_loss * np.exp(-0.15 * i) + np.random.normal(0, 5)
                elif optimizer_type == "SGD":
                    # SGD plus chaotique
                    loss = initial_loss * np.exp(-0.05 * i) + np.random.normal(0, 20)
                else:
                    # Autres optimiseurs
                    loss = initial_loss * np.exp(-0.1 * i) + np.random.normal(0, 10)
                
                loss_history.append(max(loss, 0))
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Courbe de perte
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=iterations, y=loss_history, 
                                       mode='lines+markers', name='Loss',
                                       line=dict(color='#e74c3c', width=2)))
                
                fig.update_layout(
                    title=f'Évolution de la perte - {optimizer_type}',
                    xaxis_title='Itérations',
                    yaxis_title='Loss',
                    yaxis_type="log"
                )
                
                fig = create_custom_plotly_figure(fig, f'Loss - {optimizer_type}')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Trajectoire dans l'espace des paramètres (simulation 2D)
                theta1 = np.random.normal(0, 1, n_iterations).cumsum() * 0.1
                theta2 = np.random.normal(0, 1, n_iterations).cumsum() * 0.1
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=theta1, y=theta2, 
                                       mode='lines+markers', name='Trajectoire',
                                       line=dict(color='#3498db', width=2)))
                
                fig.add_trace(go.Scatter(x=[theta1[0]], y=[theta2[0]], 
                                       mode='markers', name='Début',
                                       marker=dict(color='green', size=10)))
                
                fig.add_trace(go.Scatter(x=[theta1[-1]], y=[theta2[-1]], 
                                       mode='markers', name='Fin',
                                       marker=dict(color='red', size=10)))
                
                fig.update_layout(
                    title='Trajectoire dans l\'espace des paramètres',
                    xaxis_title='θ₁',
                    yaxis_title='θ₂'
                )
                
                fig = create_custom_plotly_figure(fig, 'Trajectoire des paramètres')
                st.plotly_chart(fig, use_container_width=True)
            
            # Métriques de convergence
            st.subheader("📊 Métriques de convergence")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Loss finale", f"{loss_history[-1]:.2f}")
            
            with metrics_col2:
                st.metric("Réduction de loss", f"{((loss_history[0] - loss_history[-1])/loss_history[0]*100):.1f}%")
            
            with metrics_col3:
                convergence_iter = next((i for i, loss in enumerate(loss_history) if loss < loss_history[0] * 0.1), n_iterations)
                st.metric("Convergence à", f"{convergence_iter} iter")
            
            with metrics_col4:
                st.metric("Learning Rate", f"{learning_rate}")
    
    with tab2:
        st.subheader("🚀 Comparaison des optimiseurs")
        
        st.markdown("""
        <div class="feature-box">
            <h4>⚡ Analyse comparative des algorithmes d'optimisation</h4>
            <p>Comparaison détaillée des performances de SGD, Adam, RMSprop, etc.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulation comparative
        optimizers = ['SGD', 'Adam', 'RMSprop', 'AdaGrad', 'Momentum']
        
        if st.button("📊 Comparer les optimiseurs"):
            fig = go.Figure()
            
            iterations = np.arange(100)
            
            # Simulation de différentes courbes de convergence
            for i, opt in enumerate(optimizers):
                if opt == 'Adam':
                    loss = 1000 * np.exp(-0.15 * iterations) + np.random.normal(0, 2, 100)
                elif opt == 'SGD':
                    loss = 1000 * np.exp(-0.03 * iterations) + np.random.normal(0, 15, 100)
                elif opt == 'RMSprop':
                    loss = 1000 * np.exp(-0.1 * iterations) + np.random.normal(0, 5, 100)
                elif opt == 'AdaGrad':
                    loss = 1000 * np.exp(-0.08 * iterations) + np.random.normal(0, 8, 100)
                else:  # Momentum
                    loss = 1000 * np.exp(-0.12 * iterations) + np.random.normal(0, 6, 100)
                
                loss = np.maximum(loss, 1)  # Éviter les valeurs négatives
                
                fig.add_trace(go.Scatter(x=iterations, y=loss, 
                                       mode='lines', name=opt,
                                       line=dict(width=2)))
            
            fig.update_layout(
                title='Comparaison des optimiseurs',
                xaxis_title='Itérations',
                yaxis_title='Loss',
                yaxis_type="log"
            )
            
            fig = create_custom_plotly_figure(fig, 'Comparaison optimiseurs')
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau de comparaison
            st.subheader("📈 Tableau comparatif")
            
            comparison_data = {
                'Optimiseur': optimizers,
                'Vitesse de convergence': ['Lente', 'Rapide', 'Moyenne', 'Moyenne', 'Rapide'],
                'Stabilité': ['Haute', 'Moyenne', 'Haute', 'Faible', 'Moyenne'],
                'Mémoire': ['Faible', 'Moyenne', 'Moyenne', 'Moyenne', 'Faible'],
                'Recommandation': ['🟡 Baseline', '🟢 Recommandé', '🟢 Bon choix', '🔴 Attention', '🟡 Classique']
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Analyse de convergence")
        
        st.markdown("""
        <div class="feature-box">
            <h4>🎯 Étude de la convergence</h4>
            <p>Analyse approfondie des critères de convergence et de la stabilité.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Paramètres d'analyse
        col1, col2 = st.columns(2)
        
        with col1:
            conv_threshold = st.slider("Seuil de convergence", 0.001, 0.1, 0.01)
            window_size = st.slider("Fenêtre d'analyse", 5, 50, 10)
        
        with col2:
            noise_level = st.slider("Niveau de bruit", 0.0, 0.5, 0.1)
            decay_rate = st.slider("Taux de décroissance", 0.01, 0.2, 0.05)
        
        # Génération de données de convergence
        n_points = 200
        x = np.arange(n_points)
        
        # Fonction de perte avec différents comportements
        base_loss = 100 * np.exp(-decay_rate * x)
        noise = np.random.normal(0, noise_level * base_loss)
        loss_with_noise = base_loss + noise
        
        # Détection de convergence
        converged_points = []
        for i in range(window_size, len(loss_with_noise)):
            window = loss_with_noise[i-window_size:i]
            if np.std(window) < conv_threshold:
                converged_points.append(i)
        
        # Graphique de convergence
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=x, y=loss_with_noise, 
                               mode='lines', name='Loss avec bruit',
                               line=dict(color='#3498db', width=1)))
        
        fig.add_trace(go.Scatter(x=x, y=base_loss, 
                               mode='lines', name='Loss théorique',
                               line=dict(color='#e74c3c', width=2, dash='dash')))
        
        if converged_points:
            convergence_point = converged_points[0]
            fig.add_vline(x=convergence_point, line_dash="dot", line_color="green",
                         annotation_text=f"Convergence à {convergence_point}")
        
        fig.update_layout(
            title='Analyse de la convergence',
            xaxis_title='Itérations',
            yaxis_title='Loss',
            yaxis_type="log"
        )
        
        fig = create_custom_plotly_figure(fig, 'Analyse de convergence')
        st.plotly_chart(fig, use_container_width=True)
        
        # Métriques de convergence
        if converged_points:
            conv_metrics_col1, conv_metrics_col2, conv_metrics_col3 = st.columns(3)
            
            with conv_metrics_col1:
                st.metric("Point de convergence", f"{converged_points[0]} iterations")
            
            with conv_metrics_col2:
                final_loss = loss_with_noise[converged_points[0]]
                st.metric("Loss à la convergence", f"{final_loss:.4f}")
            
            with conv_metrics_col3:
                stability = np.std(loss_with_noise[converged_points[0]:])
                st.metric("Stabilité post-convergence", f"{stability:.4f}")
    
    with tab4:
        st.subheader("🔬 Expérimentations")
        
        st.markdown("""
        <div class="feature-box">
            <h4>🧪 Laboratoire d'expérimentation</h4>
            <p>Zone d'expérimentation pour tester différents hyperparamètres.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration d'expérience
        st.subheader("⚙️ Configuration de l'expérience")
        
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        
        with exp_col1:
            lr_range = st.slider("Plage de Learning Rate", 0.001, 0.1, (0.01, 0.05), 0.001)
            
        with exp_col2:
            batch_sizes = st.multiselect("Tailles de batch à tester", 
                                       [8, 16, 32, 64, 128], 
                                       default=[16, 32, 64])
        
        with exp_col3:
            n_experiments = st.slider("Nombre d'expériences", 3, 10, 5)
        
        # Lancement des expériences
        if st.button("🚀 Lancer les expériences"):
            
            # Simulation de grille d'expériences
            results = []
            
            learning_rates = np.linspace(lr_range[0], lr_range[1], n_experiments)
            
            for lr in learning_rates:
                for batch_size in batch_sizes:
                    # Simulation de résultats
                    final_loss = np.random.uniform(0.1, 1.0) / lr * (batch_size / 32)
                    convergence_time = np.random.randint(50, 200)
                    stability = np.random.uniform(0.01, 0.1)
                    
                    results.append({
                        'Learning Rate': lr,
                        'Batch Size': batch_size,
                        'Final Loss': final_loss,
                        'Convergence Time': convergence_time,
                        'Stability': stability,
                        'Score': 1 / (final_loss * convergence_time * stability)
                    })
            
            results_df = pd.DataFrame(results)
            
            # Affichage des résultats
            st.subheader("📊 Résultats des expériences")
            st.dataframe(results_df.round(4), use_container_width=True)
            
            # Heatmap des résultats
            col1, col2 = st.columns(2)
            
            with col1:
                # Heatmap Learning Rate vs Batch Size pour Final Loss
                pivot_loss = results_df.pivot_table(values='Final Loss', 
                                                   index='Learning Rate', 
                                                   columns='Batch Size', 
                                                   aggfunc='mean')
                
                fig = px.imshow(pivot_loss, 
                              title="Final Loss par LR et Batch Size",
                              color_continuous_scale='RdYlBu_r')
                fig = create_custom_plotly_figure(fig, "Final Loss Heatmap")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Heatmap pour Convergence Time
                pivot_time = results_df.pivot_table(values='Convergence Time', 
                                                   index='Learning Rate', 
                                                   columns='Batch Size', 
                                                   aggfunc='mean')
                
                fig = px.imshow(pivot_time, 
                              title="Temps de convergence par LR et Batch Size",
                              color_continuous_scale='RdYlGn_r')
                fig = create_custom_plotly_figure(fig, "Convergence Time Heatmap")
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommandations automatiques
            best_config = results_df.loc[results_df['Score'].idxmax()]
            
            st.subheader("🏆 Meilleure configuration trouvée")
            
            best_col1, best_col2, best_col3, best_col4 = st.columns(4)
            
            with best_col1:
                st.metric("Learning Rate", f"{best_config['Learning Rate']:.4f}")
            
            with best_col2:
                st.metric("Batch Size", f"{int(best_config['Batch Size'])}")
            
            with best_col3:
                st.metric("Final Loss", f"{best_config['Final Loss']:.4f}")
            
            with best_col4:
                st.metric("Score", f"{best_config['Score']:.2f}")

def show_monitoring_section(df, feature_mapping):
    """Section de monitoring et métriques temps réel"""
    st.markdown('<h2 class="section-header">📈 Monitoring & Métriques</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Diagnostics", "📝 Logs"])
    
    with tab1:
        st.subheader("📊 Dashboard de monitoring")
        
        # Métriques temps réel simulées
        st.markdown("### 🚀 Métriques temps réel")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            accuracy = np.random.uniform(0.85, 0.95)
            st.metric("Accuracy", f"{accuracy:.3f}", delta=f"{np.random.uniform(-0.005, 0.005):.3f}")
        
        with col2:
            loss = np.random.uniform(0.1, 0.3)
            st.metric("Loss", f"{loss:.3f}", delta=f"{np.random.uniform(-0.01, 0.01):.3f}")
        
        with col3:
            f1_score = np.random.uniform(0.80, 0.90)
            st.metric("F1-Score", f"{f1_score:.3f}", delta=f"{np.random.uniform(-0.01, 0.01):.3f}")
        
        with col4:
            precision = np.random.uniform(0.82, 0.92)
            st.metric("Precision", f"{precision:.3f}", delta=f"{np.random.uniform(-0.01, 0.01):.3f}")
        
        with col5:
            recall = np.random.uniform(0.78, 0.88)
            st.metric("Recall", f"{recall:.3f}", delta=f"{np.random.uniform(-0.01, 0.01):.3f}")
        
        # Graphiques de monitoring
        st.markdown("### 📈 Évolution des métriques")
        
        # Génération de données temporelles
        time_points = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        # Simulation de métriques qui évoluent dans le temps
        accuracy_history = 0.7 + 0.2 * np.random.random(30).cumsum() / 30 + np.random.normal(0, 0.01, 30)
        loss_history = 1.0 * np.exp(-np.arange(30) / 10) + np.random.normal(0, 0.02, 30)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Évolution de l'accuracy
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_points, y=accuracy_history,
                                   mode='lines+markers', name='Accuracy',
                                   line=dict(color='#2ecc71', width=3)))
            
            fig.update_layout(
                title='Évolution de l\'Accuracy',
                xaxis_title='Date',
                yaxis_title='Accuracy',
                yaxis=dict(range=[0.7, 1.0])
            )
            
            fig = create_custom_plotly_figure(fig, 'Évolution Accuracy')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Évolution de la loss
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_points, y=loss_history,
                                   mode='lines+markers', name='Loss',
                                   line=dict(color='#e74c3c', width=3)))
            
            fig.update_layout(
                title='Évolution de la Loss',
                xaxis_title='Date',
                yaxis_title='Loss'
            )
            
            fig = create_custom_plotly_figure(fig, 'Évolution Loss')
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap de performance par heure
        st.markdown("### 🕐 Performance par heure de la journée")
        
        hours = np.arange(24)
        days = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
        
        # Simulation de données de performance par heure et jour
        performance_matrix = np.random.uniform(0.8, 0.95, (7, 24))
        
        fig = px.imshow(performance_matrix,
                       x=hours, y=days,
                       title="Heatmap de performance (Accuracy par heure)",
                       color_continuous_scale='RdYlGn')
        
        fig.update_layout(
            xaxis_title='Heure',
            yaxis_title='Jour de la semaine'
        )
        
        fig = create_custom_plotly_figure(fig, 'Performance Heatmap')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔍 Diagnostics avancés")
        
        st.markdown("""
        <div class="feature-box">
            <h4>🏥 État de santé du modèle</h4>
            <p>Surveillance automatique des anomalies et de la dérive des données.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Indicateurs de santé
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🎯 Stabilité du modèle")
            
            stability_score = np.random.uniform(0.85, 0.98)
            stability_color = "green" if stability_score > 0.9 else "orange" if stability_score > 0.8 else "red"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; color: white;">
                <h2 style="color: {stability_color}; margin: 0;">{stability_score:.2f}</h2>
                <p style="margin: 0;">Score de stabilité</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📊 Dérive des données")
            
            drift_score = np.random.uniform(0.05, 0.25)
            drift_color = "red" if drift_score > 0.2 else "orange" if drift_score > 0.1 else "green"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; color: white;">
                <h2 style="color: {drift_color}; margin: 0;">{drift_score:.3f}</h2>
                <p style="margin: 0;">Indice de dérive</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("#### ⚡ Performance système")
            
            system_health = np.random.uniform(0.90, 0.99)
            system_color = "green" if system_health > 0.95 else "orange" if system_health > 0.9 else "red"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; color: white;">
                <h2 style="color: {system_color}; margin: 0;">{system_health:.2f}</h2>
                <p style="margin: 0;">Santé système</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Distribution des erreurs
        st.markdown("### 📈 Analyse des erreurs")
        
        # Simulation d'erreurs
        n_samples = 1000
        errors = np.random.normal(0, 0.1, n_samples)
        predictions = np.random.uniform(0, 1, n_samples)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des erreurs
            fig = px.histogram(x=errors, nbins=50, title="Distribution des erreurs de prédiction")
            fig = create_custom_plotly_figure(fig, "Distribution des erreurs")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Erreurs vs prédictions
            fig = px.scatter(x=predictions, y=errors, 
                           title="Erreurs vs Prédictions",
                           trendline="ols")
            fig = create_custom_plotly_figure(fig, "Erreurs vs Prédictions")
            st.plotly_chart(fig, use_container_width=True)
        
        # Alertes et recommandations
        st.markdown("### 🚨 Alertes et recommandations")
        
        # Simulation d'alertes
        alerts = []
        
        if drift_score > 0.15:
            alerts.append("⚠️ Dérive des données détectée - Considérer un réentraînement")
        
        if stability_score < 0.9:
            alerts.append("🔴 Instabilité du modèle - Vérifier les hyperparamètres")
        
        if system_health < 0.95:
            alerts.append("⚡ Performance système dégradée - Optimisation recommandée")
        
        if not alerts:
            st.success("✅ Tous les indicateurs sont dans les normes")
        else:
            for alert in alerts:
                st.warning(alert)
        
        # Recommandations automatiques
        st.markdown("### 💡 Recommandations automatiques")
        
        recommendations = [
            "🔧 Ajuster le learning rate pour améliorer la convergence",
            "📊 Augmenter la taille du dataset d'entraînement",
            "🎯 Implémenter une validation croisée stratifiée",
            "⚙️ Optimiser les hyperparamètres avec Bayesian Optimization",
            "🔄 Mettre en place un pipeline de réentraînement automatique"
        ]
        
        selected_recommendations = np.random.choice(recommendations, 3, replace=False)
        
        for i, rec in enumerate(selected_recommendations, 1):
            st.info(f"{i}. {rec}")
    
    with tab3:
        st.subheader("📝 Logs et historique")
        
        st.markdown("""
        <div class="feature-box">
            <h4>📋 Journal d'activité du système</h4>
            <p>Suivi détaillé des événements et des performances.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulation de logs
        log_entries = []
        log_levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG']
        log_messages = [
            "Modèle chargé avec succès",
            "Prédiction effectuée pour utilisateur ID 12345",
            "Mise à jour des poids du modèle",
            "Sauvegarde automatique effectuée",
            "Détection d'anomalie dans les données d'entrée",
            "Optimisation des hyperparamètres terminée",
            "Évaluation des métriques complétée",
            "Nettoyage des données temporaires",
            "Nouveau batch de données traité",
            "Validation croisée lancée"
        ]
        
        # Génération de logs simulés
        current_time = pd.Timestamp.now()
        
        for i in range(50):
            timestamp = current_time - pd.Timedelta(minutes=np.random.randint(1, 1440))  # Dernières 24h
            level = np.random.choice(log_levels, p=[0.6, 0.25, 0.1, 0.05])  # Distribution réaliste
            message = np.random.choice(log_messages)
            
            log_entries.append({
                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Level': level,
                'Message': message,
                'Duration': f"{np.random.randint(10, 5000)}ms"
            })
        
        # Trier par timestamp décroissant
        log_df = pd.DataFrame(log_entries).sort_values('Timestamp', ascending=False)
        
        # Filtres pour les logs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            level_filter = st.multiselect("Filtrer par niveau:", 
                                        log_levels, 
                                        default=log_levels)
        
        with col2:
            hours_back = st.slider("Dernières X heures:", 1, 24, 6)
        
        with col3:
            search_term = st.text_input("Rechercher dans les logs:")
        
        # Application des filtres
        filtered_logs = log_df[log_df['Level'].isin(level_filter)]
        
        if search_term:
            filtered_logs = filtered_logs[filtered_logs['Message'].str.contains(search_term, case=False)]
        
        # Affichage des logs avec style conditionnel
        st.markdown("### 📊 Logs récents")
        
        # Statistiques des logs
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Total logs", len(filtered_logs))
        
        with stats_col2:
            error_count = len(filtered_logs[filtered_logs['Level'] == 'ERROR'])
            st.metric("Erreurs", error_count, delta=-np.random.randint(0, 3))
        
        with stats_col3:
            warning_count = len(filtered_logs[filtered_logs['Level'] == 'WARNING'])
            st.metric("Avertissements", warning_count, delta=-np.random.randint(0, 5))
        
        with stats_col4:
            avg_duration = filtered_logs['Duration'].str.replace('ms', '').astype(float).mean()
            st.metric("Durée moyenne", f"{avg_duration:.0f}ms")
        
        # Tableau des logs avec formatage conditionnel
        def color_logs(row):
            if row['Level'] == 'ERROR':
                return ['background-color: #ffebee'] * len(row)
            elif row['Level'] == 'WARNING':
                return ['background-color: #fff3e0'] * len(row)
            elif row['Level'] == 'INFO':
                return ['background-color: #e8f5e8'] * len(row)
            else:  # DEBUG
                return ['background-color: #f3e5f5'] * len(row)
        
        # Affichage du tableau
        st.dataframe(
            filtered_logs.head(20).style.apply(color_logs, axis=1),
            use_container_width=True,
            height=400
        )
        
        # Graphique de distribution des logs par niveau
        st.markdown("### 📊 Distribution des logs par niveau")
        
        level_counts = filtered_logs['Level'].value_counts()
        
        fig = px.pie(values=level_counts.values, 
                    names=level_counts.index,
                    title="Répartition des logs par niveau")
        
        fig = create_custom_plotly_figure(fig, "Distribution des logs")
        st.plotly_chart(fig, use_container_width=True)
        
        # Export des logs
        if st.button("📥 Exporter les logs"):
            csv = filtered_logs.to_csv(index=False)
            st.download_button(
                label="💾 Télécharger les logs (CSV)",
                data=csv,
                file_name=f"logs_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer avec informations sur l'équipe
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-top: 3rem;">
        <h3 style="font-family: 'Orbitron', monospace; margin-bottom: 1rem;">🚀 ConvexOpt Team</h3>
        <p style="font-family: 'Inter', sans-serif; margin-bottom: 0.5rem;">
            <strong>Projet d'Optimisation Convexe & Machine Learning</strong>
        </p>
        <p style="font-family: 'Inter', sans-serif; font-size: 0.9rem; opacity: 0.9;">
            🔬 Analyse théorique • 💻 Implémentation pratique • 📊 Visualisations interactives • 🎯 Interface utilisateur
        </p>
        <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">
            Développé avec ❤️ en Python | Streamlit | Plotly | Scikit-learn
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar avec informations additionnelles
def show_sidebar_info():
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
        <h4 style="margin-top: 0; font-family: 'Orbitron';">ℹ️ Info Projet</h4>
        <p style="font-size: 0.8rem; margin-bottom: 0.5rem;"><strong>Dataset:</strong> Fitness & Biométrie</p>
        <p style="font-size: 0.8rem; margin-bottom: 0.5rem;"><strong>Algorithmes:</strong> SGD, Adam, RMSprop</p>
        <p style="font-size: 0.8rem; margin-bottom: 0.5rem;"><strong>Objectif:</strong> Optimisation convexe</p>
        <p style="font-size: 0.8rem; margin-bottom: 0;"><strong>Interface:</strong> Temps réel</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Indicateurs temps réel
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #134e5e 0%, #71b280 100%); padding: 1rem; border-radius: 10px; color: white;">
        <h4 style="margin-top: 0; font-family: 'Orbitron';">📊 Status</h4>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-size: 0.8rem;">Système:</span>
            <span style="color: #2ecc71;">🟢 Actif</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-size: 0.8rem;">Modèles:</span>
            <span style="color: #f39c12;">🟡 En attente</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0;">
            <span style="font-size: 0.8rem;">API:</span>
            <span style="color: #2ecc71;">🟢 Prêt</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Raccourcis rapides
    st.sidebar.markdown("### 🚀 Raccourcis")
    
    if st.sidebar.button("🔄 Actualiser les données"):
        st.rerun()
    
    if st.sidebar.button("📊 Vue d'ensemble rapide"):
        st.session_state.quick_overview = True
    
    if st.sidebar.button("🎯 Test prédiction"):
        st.session_state.quick_prediction = True
    
    # Configuration avancée
    with st.sidebar.expander("⚙️ Configuration avancée"):
        theme = st.selectbox("Thème:", ["Sombre", "Clair", "Auto"])
        auto_refresh = st.checkbox("Actualisation automatique")
        show_debug = st.checkbox("Mode debug")
        
        if show_debug:
            st.write("🔧 Mode debug activé")
            st.write(f"Session state: {len(st.session_state)} items")

# Point d'entrée principal
if __name__ == "__main__":
    # Initialisation des états de session
    if 'quick_overview' not in st.session_state:
        st.session_state.quick_overview = False
    if 'quick_prediction' not in st.session_state:
        st.session_state.quick_prediction = False
    
    # Affichage de la sidebar
    show_sidebar_info()
    
    # Application principale
    main()
    
    # Footer
    show_footer()
    
    # Actions rapides si déclenchées
    if st.session_state.quick_overview:
        st.balloons()
        st.success("🎉 Vue d'ensemble mise à jour!")
        st.session_state.quick_overview = False
    
    if st.session_state.quick_prediction:
        st.success("🎯 Interface de prédiction prête!")
        st.session_state.quick_prediction = False