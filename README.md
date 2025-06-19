# 🎬 Movie Knowledge Graph Explorer

A comprehensive knowledge graph platform for movie data analysis using Neo4j, PyKEEN, and Streamlit. This project enables advanced graph analytics, link prediction, and knowledge graph completion for movie datasets.

## 🚀 Features

### Core Functionality
- **Interactive Web Interface**: Streamlit-based dashboard for graph exploration
- **Knowledge Graph Construction**: Automated creation of movie knowledge graphs in Neo4j
- **Graph Analytics**: Centrality analysis, community detection, and degree distribution
- **Link Prediction**: Multiple algorithms for predicting missing relationships
- **Knowledge Graph Completion**: PyKEEN integration for embedding-based completion
- **Graph Visualization**: Interactive network visualizations using PyVis

### Supported Models
- **PyKEEN Models**: TransE, RotatE, ComplEx, and other knowledge graph embedding models
- **Traditional Methods**: Jaccard similarity, Node2Vec, and machine learning approaches
- **Evaluation Metrics**: Standard ranking-based metrics (MRR, Hits@K, etc.)

## 📋 Requirements

### Dependencies
```
streamlit>=1.17.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
networkx>=2.7.0
pyvis>=0.3.0
ipython>=8.0.0
tqdm>=4.60.0
neo4j>=4.4.0
torch>=2.2.0
pykeen>=1.10.3
```

### System Requirements
- Python 3.8+
- Neo4j Database (local or remote)
- 4GB+ RAM recommended for large graphs

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd movie-knowledge-graph
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up Neo4j**
   - Install Neo4j Desktop or use Neo4j Cloud
   - Create a new database
   - Import your movie dataset (CSV files with movies, actors, directors, etc.)

## 🎯 Usage

### Launch the Web Application
```bash
streamlit run app.py
```

### Connect to Neo4j
1. Open the Streamlit interface in your browser
2. Enter your Neo4j connection details in the sidebar:
   - URI (e.g., `bolt://localhost:7687`)
   - Username (e.g., `neo4j`)
   - Password

### Explore the Graph
- **Graph Statistics**: View node counts and relationship distributions
- **Subgraph Analysis**: Extract and visualize specific subgraphs
- **Centrality Analysis**: Find influential nodes using PageRank, betweenness centrality
- **Community Detection**: Discover clusters using Louvain algorithm
- **Link Prediction**: Predict missing relationships between entities

## 📊 Core Components

### `app.py`
Main Streamlit application providing the web interface for graph exploration and analysis.

### `movie_graph.py`
Contains the `MovieKnowledgeGraph` and `MovieGraphAnalysis` classes for:
- Creating and managing the knowledge graph
- Running graph analytics algorithms
- Performing subgraph analysis

### `kg_completion.py` & `python_kg_completion.py`
PyKEEN integration for knowledge graph completion using embedding models:
- Model training and evaluation
- Triple prediction and ranking
- Standard evaluation metrics (MRR, Hits@K)

### `graph_utils.py`
Utilities for Neo4j database connections and common graph operations.

### `example_graph_queries.py`
Pre-defined Cypher queries for common graph analysis tasks.

## 🔍 Example Queries

### Find Actor Collaborations
```cypher
MATCH (a:Person {name: "Leonardo DiCaprio"})-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(costar:Person)
RETURN a.name, m.title, costar.name
```

### Get Movie Recommendations
```cypher
MATCH (m1:Movie {title: "Inception"})-[:SIMILAR_TO]-(m2:Movie)
RETURN m2.title, m2.vote_average
ORDER BY m2.vote_average DESC
```

### Community Detection
```cypher
CALL gds.louvain.stream('movieGraph')
YIELD nodeId, communityId
MATCH (n) WHERE id(n) = nodeId
RETURN communityId, count(*) AS size
ORDER BY size DESC
```

## 📈 Evaluation Metrics

The project uses standard knowledge graph completion metrics:

- **Mean Reciprocal Rank (MRR)**: Overall ranking quality
- **Hits@K**: Success rate for top-K predictions
- **Rank Percentile**: Percentage of candidates outperformed
- **Average Rank**: Mean position in candidate rankings

See `RANKING_METRICS.md` for detailed explanations.

## 🗂️ Project Structure

```
Project/
├── app.py                      # Main Streamlit application
├── movie_graph.py              # Graph creation and analysis
├── kg_completion.py            # Knowledge graph completion
├── python_kg_completion.py     # PyKEEN integration
├── graph_utils.py              # Neo4j utilities
├── example_graph_queries.py    # Sample Cypher queries
├── requirements.txt            # Python dependencies
├── RANKING_METRICS.md          # Evaluation metrics documentation
└── models/                     # Trained model storage
    ├── metadata.json
    ├── pykeen_transe_model.pkl
    └── training_triples/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyKEEN**: Knowledge graph embedding framework
- **Neo4j**: Graph database platform
- **Streamlit**: Web application framework
- **NetworkX**: Python graph library

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the project maintainers. 