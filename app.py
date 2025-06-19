import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import traceback
from IPython.display import HTML
from movie_graph import MovieKnowledgeGraph, MovieGraphAnalysis
from graph_utils import Neo4jConnector
from tqdm import tqdm

# Try to import PykeenMovieKG, but catch errors to avoid app crashes
try:
    # Force reload the module to avoid caching issues
    import importlib
    import sys
    if 'python_kg_completion' in sys.modules:
        importlib.reload(sys.modules['python_kg_completion'])
    from python_kg_completion import PykeenMovieKG
    PYKEEN_AVAILABLE = True
except Exception as e:
    import traceback
    print(f"Error importing PykeenMovieKG: {e}")
    traceback.print_exc()
    PYKEEN_AVAILABLE = False

# Callback functions for autocomplete text input
def update_head_value():
    st.session_state.head_value = st.session_state.head_input
    
def update_relation_value():
    st.session_state.relation_value = st.session_state.relation_input
    
def update_tail_value():
    st.session_state.tail_value = st.session_state.tail_input

# --- Streamlit App: Movie Knowledge Graph ---
st.set_page_config(page_title="Movie Knowledge Graph", layout="wide")
st.title("🎬 Movie Knowledge Graph Explorer")

# Initialize session state for connection status
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'connector' not in st.session_state:
    st.session_state.connector = None
if 'connection_error' not in st.session_state:
    st.session_state.connection_error = None
if 'gds_available' not in st.session_state:
    st.session_state.gds_available = False
if 'node_labels' not in st.session_state:
    st.session_state.node_labels = []
if 'relationship_types' not in st.session_state:
    st.session_state.relationship_types = []
if 'current_node_label' not in st.session_state:
    st.session_state.current_node_label = None
if 'current_node_rel_types' not in st.session_state:
    st.session_state.current_node_rel_types = []

# Sidebar: Neo4j connection
st.sidebar.header("Neo4j Connection")

def try_connect():
    uri = st.session_state.uri
    user = st.session_state.user
    password = st.session_state.password
    
    st.session_state.connector = Neo4jConnector(uri, user, password)
    try:
        # Try to run a simple test query to verify connection
        result = st.session_state.connector.execute_query("MATCH (n) RETURN count(n) AS count LIMIT 1")
        if result:
            st.session_state.connected = True
            st.session_state.connection_error = None
            # Check if GDS is available
            analyzer = MovieGraphAnalysis(st.session_state.connector)
            st.session_state.gds_available = analyzer.is_gds_available()
            
            # Get node labels and relationship types from graph statistics
            try:
                kg = MovieKnowledgeGraph(st.session_state.connector)
                stats = kg.get_graph_statistics()
                
                # Extract node labels and relationship types from statistics
                st.session_state.node_labels = list(stats['nodes'].keys())
                st.session_state.relationship_types = list(stats['relationships'].keys())
                
                # Set current node label if available
                if st.session_state.node_labels:
                    st.session_state.current_node_label = st.session_state.node_labels[0]
                    # We can't easily get relationship types for a specific node from statistics
                    # So we'll just use all relationship types for now
                    st.session_state.current_node_rel_types = st.session_state.relationship_types
            except Exception as e:
                st.warning(f"Connected successfully but could not fetch schema information: {str(e)}")
        else:
            st.session_state.connected = False
            st.session_state.connection_error = "Connection failed: Could not execute test query"
    except Exception as e:
        st.session_state.connected = False
        st.session_state.connection_error = f"Connection error: {str(e)}"

# Neo4j connection form
uri = st.sidebar.text_input("Neo4j URI", value="bolt://localhost:7687", key="uri")
user = st.sidebar.text_input("Username", value="neo4j", key="user")
password = st.sidebar.text_input("Password", type="password", key="password")
connect_btn = st.sidebar.button("Connect", on_click=try_connect)

# Display connection status
if st.session_state.connected:
    st.sidebar.success("✅ Connected to Neo4j!")
    if not st.session_state.gds_available:
        st.sidebar.warning("⚠️ Neo4j Graph Data Science library not detected. Using alternative implementations.")
elif st.session_state.connection_error:
    st.sidebar.error(st.session_state.connection_error)

def display_graph_visualization(df, source_col, target_col, weight_col=None, 
                            graph_spacing=200, repulsion_strength=10000, spring_strength=8, edge_length=300):
    """Create a graph visualization from DataFrame with source and target columns"""
    if df.empty:
        st.warning("No data to visualize")
        return
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes and edges from DataFrame
    for _, row in df.iterrows():
        source = str(row[source_col])
        target = str(row[target_col])
        weight = row[weight_col] if weight_col and weight_col in row else 1.0
        
        if source and target:
            G.add_node(source)
            G.add_node(target)
            G.add_edge(source, target, weight=weight)
    
    if len(G.nodes) == 0:
        st.warning("No valid nodes found to visualize")
        return
    
    # Create PyVis network for visualization
    net = Network(notebook=True, cdn_resources="remote", height="600px", width="100%")
    
    # Configure physics for better visualizations of crowded graphs
    # Increase repulsion between nodes and spring length for more spacing
    net.barnes_hut(
        gravity=-repulsion_strength,      # Strong negative gravity to push nodes apart
        central_gravity=0.2,              # Reduced central gravity for less clustering
        spring_length=graph_spacing,      # Controls spacing between connected nodes
        spring_strength=spring_strength / 1000.0  # Convert percentage to small decimal
    )
    
    # Add nodes with different colors based on type
    for node in G.nodes:
        # Use different colors for nodes to improve visibility
        net.add_node(node, label=node, title=node, physics=True, 
                    color="#6929c4" if node == source_col else "#1192e8")
    
    # Add edges with weight as width
    for u, v, data in G.edges(data=True):
        width = 1 + min(5, data.get('weight', 1))
        net.add_edge(u, v, width=width, title=f"Weight: {data.get('weight', 1)}")
    
    # Save visualization to temp file and display
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        net.save_graph(tmpfile.name)
        with open(tmpfile.name, 'r', encoding='utf-8') as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=600)
        # We don't delete the temp file here to avoid "file in use" errors on Windows
        # The OS will clean up the temp files automatically

# Main content
if st.session_state.connected and st.session_state.connector:
    connector = st.session_state.connector
    kg = MovieKnowledgeGraph(connector)
    analyzer = MovieGraphAnalysis(connector)

    # Helper function to update relationship types for a selected node label
    def update_rel_types_for_node():
        if st.session_state.current_node_label:
            try:
                # Get relationship types specific to the selected node label
                rel_types = analyzer.get_relationship_types_for_node(st.session_state.current_node_label)
                if rel_types:
                    st.session_state.current_node_rel_types = rel_types
                else:
                    # Fallback to all relationship types if none found for this node
                    st.session_state.current_node_rel_types = st.session_state.relationship_types
            except Exception as e:
                st.warning(f"Could not fetch relationship types: {str(e)}")
                st.session_state.current_node_rel_types = []
    
    # Helper function for node label selection change
    def on_node_label_change():
        st.session_state.current_node_label = st.session_state.node_label_select
        update_rel_types_for_node()
    
    # Helper function to refresh node labels and relationship types from graph statistics
    def refresh_schema_from_stats():
        try:
            # Get graph statistics
            stats = kg.get_graph_statistics()
            
            # Update node labels and relationship types
            st.session_state.node_labels = list(stats['nodes'].keys())
            st.session_state.relationship_types = list(stats['relationships'].keys())
            
            # Update current node's relationship types
            if st.session_state.current_node_label:
                st.session_state.current_node_rel_types = st.session_state.relationship_types
                
            return True
        except Exception as e:
            st.warning(f"Could not refresh schema information: {str(e)}")
            return False
            
    # The setup_graph functionality is now handled directly in the Knowledge Graph Completion section
    # Function to query entities for autocompletion
    def get_entities_for_autocomplete(prefix, limit=10):
        """
        Get entities that start with the given prefix for autocomplete functionality.
        
        IMPORTANT: Only returns entities that are in the trained PyKEEN model to prevent
        prediction errors. Does NOT fall back to Neo4j database queries.
        
        Args:
            prefix: The prefix to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching entity names that are in the trained model
        """
        if not prefix or len(prefix) < 2:
            return []
            
        # Only get entities from PyKEEN if it's loaded - NO Neo4j fallback
        if PYKEEN_AVAILABLE and 'pykeen_kg_instance' in st.session_state and st.session_state.pykeen_kg_instance is not None:
            pykeen_kg = st.session_state.pykeen_kg_instance
            
            # Use the new method that ensures entities are in the training set
            if hasattr(pykeen_kg, 'get_valid_entities_for_autocomplete'):
                try:
                    matching_entities = pykeen_kg.get_valid_entities_for_autocomplete(prefix, limit)
                    return matching_entities  # This method already filters and limits results
                except Exception as e:
                    print(f"Error in PyKEEN valid entity autocompletion: {e}")
            
            # Fallback to the old method but ONLY from PyKEEN entity mappings
            if hasattr(pykeen_kg, 'entity_to_id') and pykeen_kg.entity_to_id:
                try:
                    # Filter entities that start with the prefix from TRAINED MODEL ONLY
                    matching_entities = [entity for entity in pykeen_kg.entity_to_id.keys() 
                                      if entity and isinstance(entity, str) and entity.lower().startswith(prefix.lower())]
                    return sorted(matching_entities)[:limit]
                except Exception as e:
                    print(f"Error in PyKEEN entity autocompletion: {e}")
        
        # If PyKEEN is not available or has no entity mappings, return empty list
        # DO NOT query Neo4j database as it includes entities not in training set
        return []
    
    # Function to get relationship types for autocomplete
    def get_relations_for_autocomplete(prefix="", limit=20):
        """
        Get relation types for autocomplete functionality.
        
        IMPORTANT: Only returns relations that are in the trained PyKEEN model to prevent
        prediction errors. Does NOT fall back to Neo4j database queries.
        
        Args:
            prefix: Optional prefix to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of relation types that are in the trained model
        """
        # Only get relations from PyKEEN if it's loaded - NO Neo4j fallback
        if PYKEEN_AVAILABLE and 'pykeen_kg_instance' in st.session_state and st.session_state.pykeen_kg_instance is not None:
            pykeen_kg = st.session_state.pykeen_kg_instance
            
            # Use the new method that ensures relations are in the training set
            if hasattr(pykeen_kg, 'get_valid_relations_for_autocomplete'):
                try:
                    matching_relations = pykeen_kg.get_valid_relations_for_autocomplete(prefix, limit)
                    return matching_relations  # This method already filters and limits results
                except Exception as e:
                    print(f"Error in PyKEEN valid relation autocompletion: {e}")
            
            # Fallback to the old method but ONLY from PyKEEN relation mappings
            if hasattr(pykeen_kg, 'relation_to_id') and pykeen_kg.relation_to_id:
                try:
                    relations = list(pykeen_kg.relation_to_id.keys())
                    
                    # Filter by prefix if provided
                    if prefix:
                        relations = [rel for rel in relations if rel and isinstance(rel, str) and rel.lower().startswith(prefix.lower())]
                    
                    return sorted(relations)[:limit]
                except Exception as e:
                    print(f"Error in PyKEEN relation autocompletion: {e}")
        
        # If PyKEEN is not available or has no relation mappings, return empty list
        # DO NOT query Neo4j database as it includes relations not in training set
        return []

    # Main tabs for features
    tab1, tab2, tab3 = st.tabs(["Graph Statistics", "Advanced Analysis", "Run Query"])

    with tab1:
        st.subheader("Graph Statistics")
        col1, col2 = st.columns([1, 1])
        with col1:
            show_stats = st.button("Show Graph Statistics")
        with col2:
            refresh_schema = st.button("Refresh Schema Info")
        
        if refresh_schema:
            if refresh_schema_from_stats():
                st.success("Schema information refreshed successfully!")
        
        if show_stats:
            try:
                stats = kg.get_graph_statistics()
                # Create two columns for side-by-side display
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Node Counts")
                    st.dataframe(pd.DataFrame(list(stats['nodes'].items()), columns=["Node Type", "Count"]))
                with col2:
                    st.write("### Relationship Counts")
                    st.dataframe(pd.DataFrame(list(stats['relationships'].items()), columns=["Relationship Type", "Count"]))
            except Exception as e:
                st.error(f"Error retrieving graph statistics: {str(e)}")

    # Advanced Analysis tab
    with tab2:
        st.subheader("Advanced Graph Techniques")
        
        technique_options = st.radio(
            "Select Analysis Technique:",
            ["Co-occurrence Network", "Node Similarity", "Link Prediction", "Knowledge Graph Completion"]
        )
        
        if technique_options == "Co-occurrence Network":
            st.write("### Co-occurrence Network Projection")
            st.info("**Co-occurrence Networks** connect entities that share relationships with common intermediates. For example, actors who appeared in the same movies become directly connected.")
            
            # Add GDS projection management
            if st.session_state.gds_available:
                st.write("#### Manage Projections")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    show_projections = st.button("Show Projections")
                with col2:
                    delete_projections = st.button("Delete All Projections")

                if show_projections or delete_projections:
                    with st.spinner("Retrieving projections..."):
                        try:
                            projections_df = analyzer.list_gds_projections(delete=delete_projections)
                            
                            if 'message' in projections_df.columns:
                                st.info(projections_df.iloc[0]['message'])
                            elif 'error' in projections_df.columns:
                                st.error(projections_df.iloc[0]['error'])
                            elif not projections_df.empty:
                                # Display as table
                                st.write(f"Found {len(projections_df)} Network projections:")
                                st.dataframe(projections_df)
                                
                                # If projections were deleted, show success message
                                if delete_projections:
                                    st.success("Network projections have been deleted.")
                            else:
                                st.info("No projections found.")
                        except Exception as e:
                            st.error(f"Error retrieving projections: {str(e)}")
            
            st.write("#### Create New Co-occurrence Network")
            # Use dynamic node labels from database
            default_node_index = 0
            if st.session_state.node_labels:
                # Find index of "Person" if it exists, otherwise use 0
                try:
                    default_node_index = st.session_state.node_labels.index("Person")
                except ValueError:
                    default_node_index = 0
                    
            node_label = st.selectbox(
                "Node Type", 
                options=st.session_state.node_labels if st.session_state.node_labels else ["Person", "Movie", "Genre"],
                index=default_node_index,
                key="node_label_select",
                on_change=on_node_label_change
            )
            
            # Use dynamic relationship types for the selected node
            default_rel_index = 0
            if st.session_state.current_node_rel_types:
                # Find index of common relationships if they exist
                common_rels = ["ACTED_IN", "DIRECTED", "IN_GENRE"]
                for rel in common_rels:
                    try:
                        default_rel_index = st.session_state.current_node_rel_types.index(rel)
                        break
                    except ValueError:
                        continue
                        
            rel_type = st.selectbox(
                "Relationship Type", 
                options=st.session_state.current_node_rel_types if st.session_state.current_node_rel_types 
                else ["ACTED_IN", "DIRECTED", "IN_GENRE"],
                index=min(default_rel_index, len(st.session_state.current_node_rel_types)-1) if st.session_state.current_node_rel_types else 0
            )
            
            if st.session_state.gds_available:
                # Generate a descriptive projection name based on selected node and relationship
                default_proj_name = f"{node_label.lower()}_{rel_type.lower()}_cooccurrence"
                proj_name = st.text_input("Projection Name", value=default_proj_name)
                
                if st.button("Project Co-occurrence Network"):
                    with st.spinner(f"Projecting {node_label} co-occurrence network..."):
                        try:
                            analyzer.project_cooccurrence_network(
                                node_label=node_label,
                                rel_type=rel_type,
                                projected_graph_name=proj_name
                            )
                            st.success(f"Co-occurrence network projection created as '{proj_name}'")
                            st.info(f"This projection represents {node_label} nodes that share {rel_type} relationships.")
                            st.info("You can now use this projection name in the Node Similarity and Link Prediction tabs.")
                        except Exception as e:
                            st.error(f"Error creating projection: {str(e)}")
            else:
                output_rel = st.text_input("Output Relationship Type", value="CO_OCCURS_WITH")
                
                if st.button("Create Co-occurrence Network"):
                    with st.spinner(f"Creating {node_label} co-occurrence network..."):
                        try:
                            result = analyzer.create_cooccurrence_network(
                                node_label=node_label,
                                rel_type=rel_type,
                                output_rel_name=output_rel
                            )
                            st.success(result)
                        except Exception as e:
                            st.error(f"Error creating co-occurrence network: {str(e)}")
        
        elif technique_options == "Node Similarity":
            st.write("### Node Similarity Analysis")
            st.info("**Node Similarity** measures how alike nodes are based on their neighborhood structure. Uses Jaccard, Cosine, or Overlap similarity to find entities with similar connection patterns.")
            
            # Common UI components for both GDS and non-GDS versions
            algorithm = st.selectbox(
                "Similarity Algorithm", 
                ["Jaccard", "Cosine", "Overlap"],
                index=0,
                help="Jaccard: intersection/union, Cosine: dot product/product of magnitudes, Overlap: intersection/smaller set"
            )
            
            # Add top/bottom results option
            results_order = st.radio("Results Order", ["Top Results", "Bottom Results"], horizontal=True)
            
            # Create a row with input box and checkbox side by side
            col1, col2 = st.columns([3, 1])
            with col1:
                num_results = st.number_input("Number of Results", min_value=1, value=10, step=1)
            with col2:
                no_limit = st.checkbox("No Limit", value=False, help="Check to return all results without limit")
            
            # Set top_k or bottom_k based on the checkbox value and selection
            top_k = 0 if no_limit else int(num_results)
            bottom_k = 0  # Default initialization
            
            if results_order == "Bottom Results":
                bottom_k = top_k
                top_k = 0
                
            # Ensure we're actually limiting results when No Limit is not checked
            if not no_limit and top_k == 0 and bottom_k == 0:
                top_k = int(num_results)  # Default to top results if something went wrong
                
            # Add option to show graph
            show_graph = st.checkbox("Show Graph Visualization", value=True)
            
            if st.session_state.gds_available:
                # Get existing projections for dropdown
                try:
                    projections_df = analyzer.list_gds_projections(delete=False)
                    
                    # Extract projection names
                    if 'graphName' in projections_df.columns and not projections_df.empty:
                        projection_names = [name for name in projections_df['graphName'] if isinstance(name, str)]
                    else:
                        projection_names = ["cooccurrenceGraph"]
                except Exception:
                    projection_names = ["cooccurrenceGraph"]
                
                # Add a default option
                if "cooccurrenceGraph" not in projection_names:
                    projection_names.insert(0, "cooccurrenceGraph")
                
                # Create dropdown for projection name
                proj_name = st.selectbox(
                    "Projection Name", 
                    options=projection_names,
                    index=0,
                    help="Select an existing graph projection"
                )
                
                if st.button("Calculate Node Similarity"):
                    with st.spinner("Calculating node similarity..."):
                        try:
                            similarity_df = analyzer.run_node_similarity(
                                projected_graph_name=proj_name, 
                                top_k=top_k,
                                bottom_k=bottom_k,
                                algorithm=algorithm.lower()
                            )
                            if similarity_df is not None and not similarity_df.empty:
                                # Rename the column for display
                                similarity_col = f"similarity({algorithm})"
                                
                                # Handle duplicates by ensuring node1 < node2 lexicographically
                                result_type = "Bottom" if bottom_k > 0 else "Top"
                                st.write(f"### Similar Node Pairs - {algorithm} Similarity ({result_type} Results)")
                                st.dataframe(similarity_df)
                                
                                # Show visualization only if checkbox is checked
                                if show_graph:
                                    st.subheader("Similarity Network Visualization")
                                    display_graph_visualization(
                                        similarity_df, 
                                        "node1", 
                                        "node2", 
                                        similarity_col,  # Use the renamed column
                                        graph_spacing=200, 
                                        repulsion_strength=10000, 
                                        spring_strength=8, 
                                        edge_length=300
                                    )
                            else:
                                st.warning("No similarity results found. Make sure you have projected a graph first.")
                        except Exception as e:
                            st.error(f"Error calculating node similarity: {str(e)}")
            else:
                # Define default_node_index
                default_node_index = 0
                if st.session_state.node_labels:
                    # Find index of "Person" if it exists, otherwise use 0
                    try:
                        default_node_index = st.session_state.node_labels.index("Person")
                    except ValueError:
                        default_node_index = 0
                        
                node_label = st.selectbox(
                    "Node Type", 
                    options=st.session_state.node_labels if st.session_state.node_labels else ["Person", "Movie"],
                    index=min(default_node_index, len(st.session_state.node_labels)-1) if st.session_state.node_labels else 0,
                    key="similarity_node_label"
                )
                co_occurs_rel = st.text_input("Co-occurrence Relationship", value="CO_OCCURS_WITH")
                
                if st.button("Calculate Node Similarity"):
                    with st.spinner("Calculating node similarity..."):
                        try:
                            similarity_df = analyzer.calculate_node_similarity(
                                node_label=node_label,
                                co_occurs_rel=co_occurs_rel,
                                top_k=top_k,
                                bottom_k=bottom_k,
                                algorithm=algorithm.lower()
                            )
                            if not similarity_df.empty:
                                # Rename the column for display
                                similarity_col = f"similarity({algorithm})"
                                
                                # Handle duplicates by ensuring node1 < node2 lexicographically
                                result_type = "Bottom" if bottom_k > 0 else "Top"
                                st.write(f"### Similar Node Pairs - {algorithm} Similarity ({result_type} Results)")
                                st.dataframe(similarity_df)
                                
                                # Show visualization only if checkbox is checked
                                if show_graph:
                                    st.subheader("Similarity Network Visualization")
                                    display_graph_visualization(
                                        similarity_df, 
                                        "node1", 
                                        "node2", 
                                        similarity_col,  # Use the renamed column
                                        graph_spacing=200, 
                                        repulsion_strength=10000, 
                                        spring_strength=8, 
                                        edge_length=300
                                    )
                            else:
                                st.warning(f"No similarity results found. Make sure you have created {co_occurs_rel} relationships first.")
                        except Exception as e:
                            st.error(f"Error calculating node similarity: {str(e)}")
        
        elif technique_options == "Link Prediction":
            st.write("### Link Prediction for Recommendations")
            st.info("**Link Prediction** identifies missing connections in the graph using topology analysis. Predicts future collaborations, recommends similar content, and discovers hidden relationships.")
            
            algorithm = st.selectbox(
                "Select Algorithm", 
                ["Adamic-Adar", "Jaccard Similarity", "Node2Vec + ML"], 
                index=0,
                help="Different algorithms for predicting likely connections between nodes"
            )
            
            if algorithm in ["Adamic-Adar", "Jaccard Similarity"]:
                if st.session_state.gds_available:
                    # Get existing projections for dropdown
                    try:
                        projections_df = analyzer.list_gds_projections(delete=False)
                        
                        # Extract projection names
                        if 'graphName' in projections_df.columns and not projections_df.empty:
                            projection_names = [name for name in projections_df['graphName'] if isinstance(name, str)]
                        else:
                            projection_names = ["cooccurrenceGraph"]
                    except Exception:
                        projection_names = ["cooccurrenceGraph"]
                    
                    # Add a default option
                    if "cooccurrenceGraph" not in projection_names:
                        projection_names.insert(0, "cooccurrenceGraph")
                    
                    # Create dropdown for projection name
                    proj_name = st.selectbox(
                        "Projection Name", 
                        options=projection_names,
                        index=0,
                        help="Select an existing graph projection"
                    )
                    
                    # Create a row with input box and checkbox side by side
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        num_results = st.number_input("Number of Results", min_value=1, value=10, step=1)
                    with col2:
                        no_limit = st.checkbox("No Limit", value=False, help="Check to return all results without limit")
                    
                    # Set top_k based on the checkbox value
                    top_k = 0 if no_limit else int(num_results)
                    
                    if st.button("Run Link Prediction"):
                        with st.spinner(f"Running {algorithm} link prediction..."):
                            try:
                                if algorithm == "Adamic-Adar":
                                    result_df = analyzer.run_link_prediction(
                                        projected_graph_name=proj_name, 
                                        top_k=top_k
                                    )
                                else:  # Jaccard
                                    result_df = analyzer.run_jaccard_link_prediction(
                                        projected_graph_name=proj_name, 
                                        top_k=top_k
                                    )
                                    
                                if result_df is not None and not result_df.empty:
                                    # Determine what type of recommendations we're showing
                                    if "Person" in proj_name.lower() or any("person" in str(name).lower() for name in result_df["node1"].head()):
                                        rec_type = "Collaboration"
                                        st.write(f"### Top {len(result_df)} Recommended Potential Collaborations")
                                        st.info("These are suggestions for people who haven't worked together but are likely to collaborate based on their network connections.")
                                    elif "Movie" in proj_name.lower() or any("movie" in str(name).lower() for name in result_df["node1"].head()):
                                        rec_type = "Similar Movie"
                                        st.write(f"### Top {len(result_df)} Similar Movie Recommendations")
                                        st.info("These are movie pairs that aren't currently connected but are likely to be similar based on shared characteristics.")
                                    else:
                                        rec_type = "Link"
                                        st.write(f"### Top {len(result_df)} Predicted Links")
                                        st.info("These are nodes that aren't currently connected but are predicted to have a relationship.")
                                    
                                    # Add score explanation
                                    st.write("Higher scores indicate stronger prediction likelihood.")
                                    
                                    # Show the results
                                    st.dataframe(result_df)
                                    
                                    # Show visualization
                                    st.subheader(f"{rec_type} Recommendation Visualization")
                                    display_graph_visualization(
                                        result_df, 
                                        "node1", 
                                        "node2", 
                                        "score",
                                        graph_spacing=200, 
                                        repulsion_strength=10000, 
                                        spring_strength=8, 
                                        edge_length=300
                                    )
                                else:
                                    st.warning("No prediction results found. Make sure you have projected a graph first.")
                            except Exception as e:
                                st.error(f"Error running link prediction: {str(e)}")
                else:
                    # Define default_node_index
                    default_node_index = 0
                    if st.session_state.node_labels:
                        # Find index of "Person" if it exists, otherwise use 0
                        try:
                            default_node_index = st.session_state.node_labels.index("Person")
                        except ValueError:
                            default_node_index = 0
                            
                    node_label = st.selectbox(
                        "Node Type", 
                        options=st.session_state.node_labels if st.session_state.node_labels else ["Person", "Movie"],
                        index=min(default_node_index, len(st.session_state.node_labels)-1) if st.session_state.node_labels else 0,
                        key="linkpred_node_label"
                    )
                    co_occurs_rel = st.text_input("Co-occurrence Relationship", value="CO_OCCURS_WITH")
                    method = "adamic_adar" if algorithm == "Adamic-Adar" else "jaccard"
                    top_k = st.slider("Number of Results", 5, 50, 10)
                    
                    if st.button("Run Link Prediction"):
                        with st.spinner(f"Running {algorithm} link prediction..."):
                            try:
                                result_df = analyzer.calculate_link_prediction(
                                    node_label=node_label,
                                    co_occurs_rel=co_occurs_rel,
                                    method=method,
                                    top_k=top_k
                                )
                                
                                if not result_df.empty:
                                    st.write(f"### Top {len(result_df)} Predicted Links")
                                    st.dataframe(result_df)
                                    
                                    # Show visualization
                                    st.subheader("Link Prediction Visualization")
                                    display_graph_visualization(
                                        result_df, 
                                        "node1", 
                                        "node2", 
                                        "score",
                                        graph_spacing=200, 
                                        repulsion_strength=10000, 
                                        spring_strength=8, 
                                        edge_length=300
                                    )
                                else:
                                    st.warning(f"No prediction results found. Make sure you have created {co_occurs_rel} relationships first.")
                            except Exception as e:
                                st.error(f"Error running link prediction: {str(e)}")
            
            else:  # Node2Vec + ML
                st.write("#### ML-based Link Prediction")
                st.info("**Node2Vec + ML** extracts graph embeddings and trains classifiers to predict links. Provides detailed performance metrics and model interpretability.")
                
                # Get more subgraph types from graph statistics
                try:
                    graph_stats = analyzer.get_graph_statistics()
                    node_labels = list(graph_stats.get("node_counts", {}).keys())
                    relationship_types = list(graph_stats.get("relationship_counts", {}).keys())
                    
                    # Generate meaningful subgraph types
                    subgraph_options = ["Actor-Actor Network", "Movie-Movie Network"]
                    
                    # Add more options based on available node types
                    for label in node_labels:
                        if label not in ["Person", "Movie"]:  # Skip already covered types
                            subgraph_options.append(f"{label}-{label} Network")
                    
                    # Add options for bipartite networks where relevant
                    if "Person" in node_labels and "Movie" in node_labels:
                        subgraph_options.append("Person-Movie Bipartite Network")
                    
                    # Add other potential bipartite networks
                    for label1 in node_labels:
                        for label2 in node_labels:
                            if label1 != label2 and f"{label1}-{label2} Bipartite Network" not in subgraph_options:
                                for rel in relationship_types:
                                    # Check if this relationship connects these node types
                                    # This is a simplified check - in a real app, you'd query the graph
                                    if f"{label1}_{label2}" in rel or f"{label2}_{label1}" in rel:
                                        subgraph_options.append(f"{label1}-{label2} Bipartite Network")
                                        break
                except Exception as e:
                    # Fallback to default options if there's an error
                    subgraph_options = ["Actor-Actor Network", "Movie-Movie Network"]
                    print(f"Error getting graph statistics: {e}")
                
                # Select subgraph type
                subgraph_type = st.selectbox(
                    "Subgraph Type", 
                    subgraph_options, 
                    index=0
                )
                
                # Select ML algorithm
                ml_algorithm = st.selectbox(
                    "ML Algorithm",
                    ["Logistic Regression", "Random Forest", "Support Vector Machine"],
                    index=0
                )
                
                if st.button("Run ML Link Prediction"):
                    with st.spinner(f"Running {ml_algorithm} link prediction on {subgraph_type} (this may take a while)..."):
                        try:
                            # Extract subgraph based on type - with names included
                            if subgraph_type == "Actor-Actor Network":
                                query = """
                                MATCH (a:Person)-[:ACTED_IN]->(:Movie)<-[:ACTED_IN]-(b:Person)
                                WHERE id(a) < id(b)
                                RETURN id(a) AS source, id(b) AS target, a.name AS source_name, b.name AS target_name, count(*) AS weight
                                LIMIT 5000
                                """
                            elif subgraph_type == "Movie-Movie Network":
                                query = """
                                MATCH (m1:Movie)<-[:ACTED_IN]-(:Person)-[:ACTED_IN]->(m2:Movie)
                                WHERE id(m1) < id(m2)
                                RETURN id(m1) AS source, id(m2) AS target, m1.title AS source_name, m2.title AS target_name, count(*) AS weight
                                LIMIT 5000
                                """
                            elif subgraph_type == "Director-Director Network":
                                query = """
                                MATCH (d1:Person)-[:DIRECTED]->(m1:Movie)<-[:ACTED_IN]-(a:Person)-[:ACTED_IN]->(m2:Movie)<-[:DIRECTED]-(d2:Person)
                                WHERE id(d1) < id(d2) AND d1 <> d2
                                RETURN id(d1) AS source, id(d2) AS target, d1.name AS source_name, d2.name AS target_name, count(a) AS weight
                                LIMIT 5000
                                """
                            elif subgraph_type == "Person-Movie Bipartite Network":
                                query = """
                                MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
                                RETURN id(p) AS source, id(m) AS target, p.name AS source_name, m.title AS target_name, 1 AS weight
                                LIMIT 5000
                                """
                            elif "Bipartite Network" in subgraph_type:
                                # Extract node types from the selection
                                node_types = subgraph_type.split("-")
                                type1 = node_types[0].strip()
                                type2 = node_types[1].split(" ")[0].strip()
                                
                                # Generic query for bipartite networks
                                # Determine appropriate name fields for each node type
                                name_field1 = "name" if type1 in ["Person", "Director"] else "title"
                                name_field2 = "name" if type2 in ["Person", "Director"] else "title"
                                
                                query = f"""
                                MATCH (a:{type1})-[r]->(b:{type2})
                                RETURN id(a) AS source, id(b) AS target, 
                                       a.{name_field1} AS source_name, b.{name_field2} AS target_name, 
                                       1 AS weight
                                LIMIT 5000
                                """
                            else:
                                # Generic query for any other node type
                                node_parts = subgraph_type.split("-")
                                node_type = node_parts[0].strip()
                                
                                # Determine appropriate name field based on node type
                                name_field = "name" 
                                if node_type not in ["Person", "Director"]:
                                    if node_type in ["Movie", "Genre"]:
                                        name_field = "title"
                                
                                # For Director-Director, Person-Person, etc.
                                query = f"""
                                MATCH (a:{node_type})-[r1]-(shared)-[r2]-(b:{node_type})
                                WHERE id(a) < id(b)
                                RETURN id(a) AS source, id(b) AS target, 
                                       a.{name_field} AS source_name, b.{name_field} AS target_name, 
                                       count(shared) AS weight
                                LIMIT 5000
                                """

                            # Get data and build network
                            with st.spinner("Extracting subgraph from Neo4j..."):
                                df = analyzer.run_cypher_query(query)
                                if df.empty:
                                    st.error("No data returned from query")
                                    st.stop()
                                
                                # Debug info to help troubleshoot
                                st.write("Sample of extracted data:")
                                st.dataframe(df.head(3))
                                
                                import networkx as nx
                                G = nx.Graph()
                                for _, row in df.iterrows():
                                    # Make sure source_name and target_name exist
                                    source_name = str(row['source_name']) if 'source_name' in row and not pd.isna(row['source_name']) else f"Node_{row['source']}"
                                    target_name = str(row['target_name']) if 'target_name' in row and not pd.isna(row['target_name']) else f"Node_{row['target']}"
                                    
                                    # Add nodes with names as attributes
                                    G.add_node(row['source'], name=source_name)
                                    G.add_node(row['target'], name=target_name)
                                    
                                    # Add edge with weight
                                    G.add_edge(row['source'], row['target'], weight=row['weight'])
                                
                                st.write(f"Extracted graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                            
                            # Run the selected ML algorithm
                            with st.spinner(f"Running {ml_algorithm} classification..."):
                                y_true, y_scores, predictions_df = analyzer.run_ml_link_prediction(
                                    G, algorithm=ml_algorithm.lower().replace(" ", "_")
                                )
                                metrics = analyzer.evaluate_link_prediction(y_true, y_scores)
                                
                                # Display the top predicted links
                                if not predictions_df.empty:
                                    # Determine the recommendation type based on subgraph type
                                    if "Person" in subgraph_type or "Actor" in subgraph_type or "Director" in subgraph_type:
                                        rec_type = "Collaboration"
                                        rec_explanation = "potential collaborations between people"
                                    elif "Movie" in subgraph_type:
                                        rec_type = "Similar Movie"
                                        rec_explanation = "similar movies"
                                    else:
                                        rec_type = "Link"
                                        rec_explanation = "predicted connections"
                                        
                                    st.write(f"### {rec_type} Recommendations")
                                    st.info(f"The ML model has learned patterns from the existing connections and is now suggesting {rec_explanation} that don't currently exist.")
                                    
                                    # Create tabs for different views of the predictions
                                    pred_tabs = st.tabs(["Top Recommendations", "Model Evaluation", "All Predictions"])
                                    
                                    with pred_tabs[0]:
                                        # Show top 15 predictions by score
                                        st.write(f"#### Top 15 {rec_type} Recommendations")
                                        st.info(f"These are the most likely {rec_explanation} according to the ML model. Higher probability means stronger likelihood of the recommendation.")
                                        
                                        # Get top 15 predictions sorted by score
                                        top_preds = predictions_df.sort_values('score', ascending=False).head(15).copy()
                                        
                                        # Format the score as percentage
                                        top_preds['probability'] = top_preds['score'].apply(lambda x: f"{x:.2%}")
                                        
                                        # Count how many are true links
                                        true_count = top_preds['is_true_link'].sum()
                                        st.write(f"Among top 15 predictions: {true_count} are actual connections, {15-true_count} are new recommendations.")
                                        
                                        st.dataframe(
                                            top_preds[['node1', 'node2', 'probability', 'is_true_link']]
                                            .rename(columns={'is_true_link': 'Already Exists (1=yes)'})
                                        )
                                        
                                    with pred_tabs[1]:
                                        # Show model evaluation metrics and confusion matrix
                                        st.write("#### Model Evaluation")
                                        st.info("This tab shows the model's performance metrics and confusion matrix to help evaluate prediction accuracy.")
                                        
                                        # Create two columns for metrics and confusion matrix
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            # Show performance metrics
                                            st.write("##### Performance Metrics")
                                            metrics_df = pd.DataFrame({
                                                'Metric': list(metrics.keys()),
                                                'Value': list(metrics.values())
                                            })
                                            st.dataframe(metrics_df)
                                        
                                        with col2:
                                            # Show confusion matrix
                                            st.write("##### Confusion Matrix")
                                            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                                            import matplotlib.pyplot as plt
                                            
                                            # Calculate confusion matrix
                                            y_pred = (np.array(y_scores) >= 0.5).astype(int)
                                            cm = confusion_matrix(y_true, y_pred)
                                            
                                            # Create confusion matrix plot
                                            fig, ax = plt.subplots(figsize=(6, 4))
                                            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Link', 'Link'])
                                            disp.plot(cmap='Blues', ax=ax)
                                            plt.title('Confusion Matrix')
                                            st.pyplot(fig)
                                            
                                            # Add explanation
                                            st.write("""
                                            **Confusion Matrix Explanation:**
                                            - True Negatives (Top Left): Correctly predicted non-links
                                            - False Positives (Top Right): Incorrectly predicted links
                                            - False Negatives (Bottom Left): Missed actual links
                                            - True Positives (Bottom Right): Correctly predicted links
                                            """)
                                        
                                    with pred_tabs[2]:
                                        # Show all predictions
                                        st.write(f"#### All {rec_type} Predictions")
                                        st.info("This tab shows all predictions made by the model, sorted by probability.")
                                        
                                        # Display all predictions
                                        all_preds = predictions_df.copy().sort_values('score', ascending=False)
                                        all_preds['probability'] = all_preds['score'].apply(lambda x: f"{x:.2%}")
                                        
                                        # Count new recommendations
                                        new_recs = len(all_preds[all_preds['is_true_link'] == 0])
                                        existing = len(all_preds[all_preds['is_true_link'] == 1])
                                        
                                        st.write(f"Showing all {len(all_preds)} predictions: {new_recs} new recommendations and {existing} existing connections")
                                        st.dataframe(
                                            all_preds[['node1', 'node2', 'probability', 'is_true_link']]
                                            .rename(columns={'is_true_link': 'Already Exists (1=yes)'})
                                        )
                                    
                                    # Show visualization of top predictions
                                    st.write(f"### {rec_type} Recommendation Visualization")
                                    st.write(f"Visualizing top 15 {rec_explanation}:")
                                    viz_df = predictions_df.sort_values('score', ascending=False).head(15).copy()  # Using same number (15) as the table
                                    viz_df['score'] = viz_df['score'] * 10  # Scale up for better visualization
                                    display_graph_visualization(
                                        viz_df, 
                                        "node1", 
                                        "node2", 
                                        "score",
                                        graph_spacing=200, 
                                        repulsion_strength=10000, 
                                        spring_strength=8, 
                                        edge_length=300
                                    )
                                
                                # Show ROC curve
                                from sklearn.metrics import roc_curve, auc
                                import matplotlib.pyplot as plt
                                import numpy as np
                                
                                fpr, tpr, _ = roc_curve(y_true, y_scores)
                                roc_auc = auc(fpr, tpr)
                                
                                fig, ax = plt.subplots(figsize=(6, 6))
                                ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                                ax.plot([0, 1], [0, 1], 'k--', lw=2)
                                ax.set_xlim([0.0, 1.0])
                                ax.set_ylim([0.0, 1.05])
                                ax.set_xlabel('False Positive Rate')
                                ax.set_ylabel('True Positive Rate')
                                ax.set_title(f'ROC Curve - {ml_algorithm} Link Prediction on {subgraph_type}')
                                ax.legend(loc="lower right")
                                st.pyplot(fig)
                                
                        except Exception as e:
                            st.error(f"Error running Node2Vec link prediction: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        
        elif technique_options == "Knowledge Graph Completion":
            st.write("### Knowledge Graph Completion with PyKEEN")

            # Check if PyKEEN is available before trying to use it
            if not PYKEEN_AVAILABLE:
                st.error("PyKEEN is not available. There appears to be a compatibility issue with PyTorch.")
                st.warning("Possible solutions:\n1. Update PyKEEN to a compatible version with: `pip install pykeen>=1.10.3`\n2. Install all dependencies with: `pip install -r requirements.txt`")
                st.stop()
                
            # Instantiate PykeenMovieKG if not already done (e.g., at app start or connection)
            if 'pykeen_kg_instance' not in st.session_state or st.session_state.pykeen_kg_instance is None:
                try:
                    # Force reload to ensure we're using the latest implementation
                    import importlib
                    import sys
                    if 'python_kg_completion' in sys.modules:
                        importlib.reload(sys.modules['python_kg_completion'])
                    from python_kg_completion import PykeenMovieKG
                        
                    # Create a fresh instance
                    st.session_state.pykeen_kg_instance = PykeenMovieKG(connector)
                    
                    # If a model exists for the selected model type, try to load it to get entity mappings
                    for model_name in ["TransE", "PairRE", "RotatE"]:
                        model_file = f'models/pykeen_{model_name.lower()}_model.pkl'
                        if os.path.exists(model_file):
                            try:
                                st.session_state.pykeen_kg_instance.load_model(model_name)
                                st.success(f"Loaded {model_name} model for entity suggestions")
                                break
                            except Exception:
                                continue
                    
                    # If no model was loaded, try to prepare training data to get entities
                    if not st.session_state.pykeen_kg_instance.entity_to_id:
                        st.session_state.pykeen_kg_instance.prepare_training_data()
                        
                    st.info("PykeenMovieKG helper initialized.")
                except Exception as e:
                    st.error(f"Failed to initialize PykeenMovieKG: {e}")
                    st.code(traceback.format_exc())
                    st.stop()
            else:
                # Check if the instance has the required attributes
                if not hasattr(st.session_state.pykeen_kg_instance, 'model_name'):
                    # Reinitialize with the proper class
                    try:
                        # Force reload to ensure we're using the latest implementation
                        import importlib
                        import sys
                        if 'python_kg_completion' in sys.modules:
                            importlib.reload(sys.modules['python_kg_completion'])
                        from python_kg_completion import PykeenMovieKG
                            
                        # Create a fresh instance
                        st.session_state.pykeen_kg_instance = PykeenMovieKG(connector)
                        
                        # If a model exists for the selected model type, try to load it to get entity mappings
                        for model_name in ["TransE", "PairRE", "RotatE"]:
                            model_file = f'models/pykeen_{model_name.lower()}_model.pkl'
                            if os.path.exists(model_file):
                                try:
                                    st.session_state.pykeen_kg_instance.load_model(model_name)
                                    st.success(f"Loaded {model_name} model for entity suggestions")
                                    break
                                except Exception:
                                    continue
                        
                        # If no model was loaded, try to prepare training data to get entities
                        if not st.session_state.pykeen_kg_instance.entity_to_id:
                            st.session_state.pykeen_kg_instance.prepare_training_data()
                            
                        st.info("PykeenMovieKG helper reinitialized.")
                    except Exception as e:
                        st.error(f"Failed to reinitialize PykeenMovieKG: {e}")
                        st.code(traceback.format_exc())
                        st.stop()
                        
            pykeen_kg = st.session_state.pykeen_kg_instance
            
            # Add explanation with examples
            with st.expander("What is Knowledge Graph Completion?", expanded=True):
                st.markdown("""
                **Knowledge Graph Completion** predicts missing relationships in knowledge graphs using machine learning embeddings.
                
                **Key Applications:**
                - 🎬 **Movie Recommendations**: Find similar films or potential collaborations
                - 🤝 **Actor-Director Matching**: Predict likely future partnerships
                - 🔗 **Link Prediction**: Discover missing connections between entities
                - 📊 **Similarity Analysis**: Find semantically related entities
                
                **Available Models:**
                - **TransE**: Simple translational model (h + r ≈ t). Fast, interpretable.
                - **RotatE**: Rotation-based model for complex relationships. Handles patterns like symmetry/antisymmetry.
                - **PairRE**: Handles multiple relation types and N-to-N relationships effectively.
                
                **How it works**: Models learn vector representations of entities and relations, then predict missing triples by measuring compatibility in embedding space.
                """)
            
            # Create the two-column layout for completion controls
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Simplified method selection with better labels
                pykeen_model_options = ["TransE", "RotatE", "PairRE"] # Add more if supported by your class
                selected_pykeen_model = st.selectbox(
                    "Select PyKEEN Model", 
                    pykeen_model_options, 
                    index=0,
                    help="Select the PyKEEN model to train and use for completion."
                )
                
                # Training parameters
                st.subheader(f"Train {selected_pykeen_model} Model")
                training_epochs = st.number_input("Training Epochs", min_value=1, value=100, step=10, key=f"epochs_{selected_pykeen_model.lower()}")
                embedding_dim = st.number_input("Embedding Dimensions", min_value=10, value=100, step=10, key=f"emb_dim_{selected_pykeen_model.lower()}")
                device_option = st.selectbox("Device", ["cpu", "cuda"], index=0, key=f"device_{selected_pykeen_model.lower()}")

            with col2:
                st.write(f"#### Status for {selected_pykeen_model}")
                model_file = f'models/pykeen_{selected_pykeen_model.lower()}_model.pkl'
                if os.path.exists(model_file):
                    st.success(f"✅ Trained {selected_pykeen_model} model found!")
                    try:
                        file_size = os.path.getsize(model_file) / (1024 * 1024)  # Size in MB
                        st.write(f"Model size: {file_size:.2f} MB")
                    except Exception as e:
                        st.write(f"Could not get model size: {e}")
                else:
                    st.warning(f"⚠️ No trained {selected_pykeen_model} model found at {model_file}.")
                
                st.info(f"{selected_pykeen_model} is a KGE model. Training involves learning embeddings for entities and relations. This can take time.")


            if st.button(f"Train {selected_pykeen_model} Model Now", key=f"train_btn_{selected_pykeen_model.lower()}"):
                with st.spinner(f"Training {selected_pykeen_model} model (epochs: {training_epochs}, dim: {embedding_dim})... This may take a while."):
                    try:
                        # Get the pykeen_kg instance from session state
                        if 'pykeen_kg_instance' not in st.session_state or st.session_state.pykeen_kg_instance is None:
                            st.error("PyKEEN instance not found. Please refresh the page and try again.")
                            st.stop()
                            
                        # Use the pykeen_kg from session state
                        pykeen_kg = st.session_state.pykeen_kg_instance
                            
                        # Make sure the models directory exists
                        os.makedirs("models", exist_ok=True)
                        
                        # Prepare training data (loads triples and creates factory)
                        # This should be called before each training if factory might change or not exist
                        if not hasattr(pykeen_kg, 'factory') or pykeen_kg.factory is None: # Check if factory exists, if not, prepare_training_data
                             st.write("Preparing training data (loading triples)...")
                             if not pykeen_kg.prepare_training_data():
                                 st.error("Failed to prepare training data. Cannot train model.")
                                 st.stop()
                             st.write("Training data prepared.")
                        
                        # Train the model
                        trained_model = pykeen_kg.train_model(
                            model_name=selected_pykeen_model,
                            epochs=training_epochs,
                            embedding_dim=embedding_dim,
                            device=device_option,
                            save_model_flag=True, # Ensure it's saved
                            model_output_dir="models"
                        )
                        
                        if trained_model:
                            st.success(f"✅ {selected_pykeen_model} model trained successfully!")
                            st.balloons()
                            # Rerun to update model status
                            st.rerun()
                        else:
                            st.error(f"Error training {selected_pykeen_model} model. Check console/logs for details.")
                    except Exception as e:
                        st.error(f"Exception during model training: {str(e)}")
                        st.code(traceback.format_exc())
            
            st.markdown("---")
            st.write("#### Link Prediction / Completion (using trained models)")
            
            # Check if the model exists for the selected model type
            if os.path.exists(model_file):
                # Add tabs for different prediction types
                pred_tabs = st.tabs(["Triple Prediction", "Movie Recommendation"])
                
                with pred_tabs[0]:
                    st.subheader("Predict Missing Parts of Knowledge Graph Triples")
                    st.write("Fill in two parts of a triple (head, relation, tail) and predict the missing part.")
                    
                    # Add helpful note about entity availability
                    st.info("💡 **Important:** The autocomplete suggestions show only entities that were included in the model training data. If you don't see an entity you're looking for, it means it wasn't part of the training subset. This prevents prediction errors.")
                    
                    # Create a form-like UI with three columns
                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    
                    with pred_col1:
                        predict_head = st.checkbox("Predict Head", value=False, key="predict_head")
                        if predict_head:
                            head_value = None
                        else:
                            head_value = st.text_input("Head Entity", key="head_input", on_change=update_head_value)
                            # Add autocomplete option
                            if head_value and len(head_value) >= 2:
                                suggestions = get_entities_for_autocomplete(head_value)
                                if suggestions:
                                    selected_suggestion = st.selectbox(
                                        "Select from suggestions", 
                                        options=[""] + suggestions,
                                        key="head_suggestion"
                                    )
                                    if selected_suggestion:
                                        head_value = selected_suggestion
                    
                    with pred_col2:
                        predict_relation = st.checkbox("Predict Relation", value=False, key="predict_relation")
                        if predict_relation:
                            relation_value = None
                        else:
                            relation_value = st.text_input("Relation Type", key="relation_input", on_change=update_relation_value)
                            # Add relation autocomplete
                            if relation_value and len(relation_value) >= 1:
                                rel_suggestions = get_relations_for_autocomplete(relation_value)
                                if rel_suggestions:
                                    selected_rel = st.selectbox(
                                        "Select from relation types", 
                                        options=[""] + rel_suggestions,
                                        key="relation_suggestion"
                                    )
                                    if selected_rel:
                                        relation_value = selected_rel
                    
                    with pred_col3:
                        predict_tail = st.checkbox("Predict Tail", value=False, key="predict_tail")
                        if predict_tail:
                            tail_value = None
                        else:
                            tail_value = st.text_input("Tail Entity", key="tail_input", on_change=update_tail_value)
                            # Add autocomplete option
                            if tail_value and len(tail_value) >= 2:
                                tail_suggestions = get_entities_for_autocomplete(tail_value)
                                if tail_suggestions:
                                    selected_tail = st.selectbox(
                                        "Select from suggestions", 
                                        options=[""] + tail_suggestions,
                                        key="tail_suggestion"
                                    )
                                    if selected_tail:
                                        tail_value = selected_tail
                    
                    # Validate that exactly one of the three is None
                    prediction_ready = sum(x is None for x in [head_value, relation_value, tail_value]) == 1
                    
                    # Number of predictions to show
                    k_predictions = st.slider("Number of predictions", min_value=1, max_value=20, value=5)
                    
                    if not prediction_ready:
                        st.warning("Please check exactly one of the 'Predict' options and fill in the other two fields.")
                    
                    # Run prediction button
                    if prediction_ready and st.button("Run Prediction", key="run_triple_prediction"):
                        with st.spinner(f"Loading {selected_pykeen_model} model and predicting..."):
                            try:
                                # Make sure we have a PykeenMovieKG instance
                                if 'pykeen_kg_instance' not in st.session_state or st.session_state.pykeen_kg_instance is None:
                                    st.error("PyKEEN instance not found. Please refresh the page and try again.")
                                    st.stop()
                                
                                pykeen_kg = st.session_state.pykeen_kg_instance
                                
                                # Load the model if not already loaded
                                if pykeen_kg.model_name != selected_pykeen_model or pykeen_kg.model is None:
                                    if not pykeen_kg.load_model(selected_pykeen_model):
                                        st.error(f"Failed to load {selected_pykeen_model} model. Please try training again.")
                                        st.stop()
                                
                                # Run prediction
                                predictions = pykeen_kg.predict_triplet(
                                    head=head_value,
                                    relation=relation_value, 
                                    tail=tail_value,
                                    k=k_predictions
                                )
                                
                                if "Error" in predictions.columns:
                                    st.error(f"Prediction error: {predictions['Error'][0]}")
                                elif not predictions.empty:
                                    st.success(f"Found {len(predictions)} predictions")
                                    
                                    # Display as table
                                    st.dataframe(predictions)
                                else:
                                    st.warning("No predictions found.")
                            except Exception as e:
                                st.error(f"Error during prediction: {str(e)}")
                                st.code(traceback.format_exc())
                
                with pred_tabs[1]:
                    st.subheader("Find Similar Movies")
                    st.info("**Find Similar Movies** uses embedding similarity to discover movies with comparable characteristics. The model analyzes learned patterns to suggest movies that share similar features in the embedding space.")
                    
                    # Number of recommendations
                    k_recommend = st.slider("Number of recommendations", min_value=1, max_value=50, value=10, key="k_recommend")
                    
                    if st.button("Find Similar Movies", key="find_similar_movies"):
                        with st.spinner(f"Loading {selected_pykeen_model} model and finding similar movies..."):
                            try:
                                # Make sure we have a PykeenMovieKG instance
                                if 'pykeen_kg_instance' not in st.session_state or st.session_state.pykeen_kg_instance is None:
                                    st.error("PyKEEN instance not found. Please refresh the page and try again.")
                                    st.stop()
                                
                                pykeen_kg = st.session_state.pykeen_kg_instance
                                
                                # Load the model if not already loaded
                                if pykeen_kg.model_name != selected_pykeen_model or pykeen_kg.model is None:
                                    if not pykeen_kg.load_model(selected_pykeen_model):
                                        st.error(f"Failed to load {selected_pykeen_model} model. Please try training again.")
                                        st.stop()
                                
                                # Run movie similarity search
                                recommendations = pykeen_kg.find_similar_movies(
                                    k=k_recommend
                                )
                                
                                if "Error" in recommendations.columns:
                                    st.error(f"Error finding similar movies: {recommendations['Error'][0]}")
                                elif not recommendations.empty:
                                    # Show which relationships were used
                                    if "Relations Used" in recommendations.columns:
                                        relations_used = recommendations["Relations Used"].iloc[0] if len(recommendations) > 0 else 0
                                        st.success(f"Found {len(recommendations)} similar movies using {relations_used} relationship types")
                                    else:
                                        st.success(f"Found {len(recommendations)} similar movies")
                                    
                                    # Display as table (hide Relations Used column if present)
                                    display_cols = ["Source Movie", "Target Entity", "Similarity"]
                                    if all(col in recommendations.columns for col in display_cols):
                                        st.dataframe(recommendations[display_cols])
                                    else:
                                        st.dataframe(recommendations)
                                else:
                                    st.warning("No similar movies found.")
                            except Exception as e:
                                st.error(f"Error finding similar movies: {str(e)}")
                                st.code(traceback.format_exc())
            else:
                st.warning(f"No trained model found for {selected_pykeen_model}. Please train the model first.")
                
            st.markdown("---")

    # Run Query tab
    with tab3:
        st.subheader("Run Custom Cypher Query")
        
        # Add example queries section
        with st.expander("Example Queries (Click to Use)", expanded=True):
            # Create a list of example queries with descriptions
            example_queries = [
                # Basic Search Queries
                {"name": "Movie Search", "description": "Find movies containing a specific title", 
                 "query": """MATCH (m:Movie)
                WHERE m.title CONTAINS "Matrix"
                RETURN m.title, m.release_year, m.vote_average
                LIMIT 10"""},
                
                {"name": "Actor-Movie Graph", "description": "Show actors and their movies (visualize as graph)", 
                 "query": """MATCH path = (p:Person)-[:ACTED_IN]->(m:Movie)
                WHERE m.title CONTAINS "Matrix"
                RETURN path LIMIT 25"""},
                
                # Exploratory Graph Analysis - Degree Distribution
                {"name": "Actor Degree Distribution", "description": "Analyze how many movies each actor has been in", 
                 "query": """MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
                WITH p, count(m) AS degree
                RETURN degree, count(p) AS actor_count
                ORDER BY degree DESC
                LIMIT 20"""},
                
                {"name": "Movie Cast Size Distribution", "description": "Distribution of cast sizes across movies", 
                 "query": """MATCH (m:Movie)<-[:ACTED_IN]-(p:Person)
                WITH m, count(p) AS cast_size
                RETURN cast_size, count(m) AS movie_count
                ORDER BY cast_size DESC
                LIMIT 20"""},
                
                {"name": "Top Connected People", "description": "Find most connected people in the graph", 
                 "query": """MATCH (p:Person)-[r]-()
                WITH p, count(r) AS connections
                RETURN p.name AS person, connections
                ORDER BY connections DESC
                LIMIT 15"""},
                
                # Centrality Analysis
                {"name": "Actor Centrality Analysis", "description": "Find actors who connect different parts of the graph", 
                 "query": """MATCH (p:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(co:Person)
                WHERE p <> co
                WITH p, count(DISTINCT co) AS co_actors, count(DISTINCT m) AS movies
                RETURN p.name AS actor, co_actors, movies, 
                       round(toFloat(co_actors)/movies, 2) AS centrality_score
                ORDER BY centrality_score DESC
                LIMIT 15"""},
                
                {"name": "Genre Bridge Analysis", "description": "Find actors who work across multiple genres", 
                 "query": """MATCH (p:Person)-[:ACTED_IN]->(m:Movie)-[:IN_GENRE]->(g:Genre)
                WITH p, collect(DISTINCT g.name) AS genres, count(DISTINCT m) AS movies
                WHERE size(genres) >= 3
                RETURN p.name AS actor, genres, movies, size(genres) AS genre_diversity
                ORDER BY genre_diversity DESC, movies DESC
                LIMIT 15"""},
                
                # Community Detection
                {"name": "Genre Communities", "description": "Find clusters of related genres", 
                 "query": """MATCH (g1:Genre)<-[:IN_GENRE]-(m:Movie)-[:IN_GENRE]->(g2:Genre)
                WHERE g1.name < g2.name
                WITH g1, g2, count(m) AS shared_movies
                WHERE shared_movies >= 5
                RETURN g1.name AS genre1, g2.name AS genre2, shared_movies
                ORDER BY shared_movies DESC
                LIMIT 20"""},
                
                {"name": "Director-Actor Communities", "description": "Find frequent director-actor collaborations", 
                 "query": """MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person)
                WITH d, a, count(m) AS collaborations
                WHERE collaborations >= 2
                RETURN d.name AS director, a.name AS actor, collaborations
                ORDER BY collaborations DESC
                LIMIT 20"""},
                
                # Subgraph Analysis - Actor Networks
                {"name": "Actor Collaboration Network", "description": "Network of actors who worked together", 
                 "query": """MATCH (a1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(a2:Person)
                WHERE a1.name CONTAINS "Tom Hanks" AND a1 <> a2
                WITH a1, a2, collect(m.title) AS movies, count(m) AS collaborations
                RETURN a1.name AS actor1, a2.name AS actor2, movies, collaborations
                ORDER BY collaborations DESC
                LIMIT 15"""},
                
                {"name": "Extended Actor Network", "description": "2-hop actor network (actors of actors)", 
                 "query": """MATCH path = (a1:Person)-[:ACTED_IN]->(m1:Movie)<-[:ACTED_IN]-(a2:Person)-[:ACTED_IN]->(m2:Movie)<-[:ACTED_IN]-(a3:Person)
                WHERE a1.name CONTAINS "Keanu Reeves" AND a1 <> a2 AND a2 <> a3 AND a1 <> a3
                RETURN path
                LIMIT 30"""},
                
                {"name": "Director's Actor Network", "description": "Network of actors who worked with the same director", 
                 "query": """MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person)
                WHERE d.name CONTAINS "Christopher Nolan"
                WITH d, collect(DISTINCT a) AS actors, collect(DISTINCT m.title) AS movies
                UNWIND actors AS a1
                UNWIND actors AS a2
                WITH d, a1, a2, movies WHERE a1 <> a2
                RETURN d.name AS director, a1.name AS actor1, a2.name AS actor2, movies
                LIMIT 25"""},
                
                # Subgraph Analysis - Genre-based Groupings
                {"name": "Genre-based Movie Network", "description": "Movies connected through shared genres", 
                 "query": """MATCH (m1:Movie)-[:IN_GENRE]->(g:Genre)<-[:IN_GENRE]-(m2:Movie)
                WHERE g.name = "Action" AND m1 <> m2
                WITH m1, m2, collect(g.name) AS shared_genres, count(g) AS genre_overlap
                WHERE genre_overlap >= 2
                RETURN m1.title AS movie1, m2.title AS movie2, shared_genres, genre_overlap
                ORDER BY genre_overlap DESC
                LIMIT 20"""},
                
                {"name": "Genre Evolution Over Time", "description": "How genre popularity changed over decades", 
                 "query": """MATCH (g:Genre)<-[:IN_GENRE]-(m:Movie)
                WHERE m.release_year >= 1990 AND m.release_year <= 2020
                WITH g.name AS genre, (m.release_year/10)*10 AS decade, count(m) AS movies
                RETURN genre, decade, movies
                ORDER BY genre, decade"""},
                
                {"name": "Multi-Genre Movie Analysis", "description": "Movies that span multiple genres", 
                 "query": """MATCH (m:Movie)-[:IN_GENRE]->(g:Genre)
                WITH m, collect(g.name) AS genres, count(g) AS genre_count
                WHERE genre_count >= 3
                RETURN m.title AS movie, m.release_year AS year, genres, genre_count
                ORDER BY genre_count DESC, m.vote_average DESC
                LIMIT 20"""},
                
                # Path Analysis
                {"name": "Actor Path Finder", "description": "Find the shortest path between actors (like Six Degrees of Kevin Bacon)", 
                 "query": """MATCH path = shortestPath((p1:Person)-[:ACTED_IN*]-(p2:Person))
                WHERE p1.name CONTAINS "Keanu Reeves" AND p2.name CONTAINS "Tom Hanks"
                RETURN path"""},
                
                {"name": "Movie Relationship Paths", "description": "Find connection paths between movies", 
                 "query": """MATCH path = shortestPath((m1:Movie)-[:ACTED_IN|DIRECTED|IN_GENRE*..4]-(m2:Movie))
                WHERE m1.title CONTAINS "Matrix" AND m2.title CONTAINS "Inception"
                RETURN path
                LIMIT 5"""},
                
                # Visualization Queries
                {"name": "Movie Community", "description": "Visualize a movie and all its connections (graph visualization)", 
                 "query": """MATCH path = (m:Movie)-[r:ACTED_IN|DIRECTED|IN_GENRE|PRODUCED_BY]-(n)
                WHERE m.title CONTAINS "Matrix"
                RETURN path LIMIT 50"""},
                
                {"name": "Director's Universe", "description": "Complete network around a director", 
                 "query": """MATCH path = (d:Person)-[:DIRECTED]->(m:Movie)-[r:ACTED_IN|IN_GENRE]-(n)
                WHERE d.name CONTAINS "Quentin Tarantino"
                RETURN path
                LIMIT 100"""}
            ]
            
            # Create a DataFrame from the examples for nice display
            df = pd.DataFrame(example_queries)
            
            # Function for clicking on a query example
            if 'current_query' not in st.session_state:
                st.session_state.current_query = example_queries[0]["query"]
                
            def select_query(query_text):
                st.session_state.current_query = query_text
            
            # Display examples as a clickable table
            for i, row in df.iterrows():
                col1, col2, col3 = st.columns([2, 5, 1])
                with col1:
                    st.markdown(f"**{row['name']}**")
                with col2:
                    st.write(row['description'])
                with col3:
                    st.button(f"Use", key=f"use_query_{i}", on_click=select_query, args=(row['query'],))
                    
        # Query input
        query = st.text_area("Enter Cypher Query", 
                           height=150,
                           value=st.session_state.current_query)
        
        # Display options
        display_mode = st.radio("Display Results As:", ["Table", "Graph"], index=0)
        
        # Configure graph visualization if graph mode is selected
        if display_mode == "Graph":
            st.info("For graph visualization, your query must return full nodes and relationships, not just properties.")
            
            # Add graph layout options
            with st.expander("Graph Layout Options", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    graph_spacing = st.slider("Node Spacing", 100, 500, 200, 50, 
                                            help="Increase for more space between nodes (helps with crowded graphs)")
                    edge_length = st.slider("Edge Length", 100, 500, 300, 50,
                                          help="Length of the connection lines between nodes")
                with col2:
                    repulsion_strength = st.slider("Repulsion Strength", 1000, 20000, 10000, 1000, 
                                                 help="Higher values push nodes further apart")
                    spring_strength = st.slider("Spring Strength", 1, 100, 8, 1, 
                                              help="Lower values (as percentage) make connections more flexible", 
                                              format="%d%%")
            
            graph_query_example = """
            MATCH path = (m:Movie)-[:ACTED_IN|DIRECTED]-(p:Person)
            WHERE m.title CONTAINS "Matrix"
            RETURN path LIMIT 50
            """
            
            if "RETURN" in query and not any(x in query.upper() for x in ["RETURN path", "RETURN m,", "RETURN n,"]):
                st.warning("Your query may not return full nodes/relationships. For graph visualization, use queries that return paths or full nodes.")
                if st.button("Use Example Graph Query"):
                    query = graph_query_example
                    # Force page refresh to show new query
                    st.rerun()
            
        # Run button
        if st.button("Run Query"):
            try:
                with st.spinner("Executing query..."):
                    result = connector.execute_query(query)
                
                # Display results
                if not result:
                    st.warning("Query returned no results")
                elif display_mode == "Table":
                    # Convert Neo4j result to DataFrame
                    data = []
                    for record in result:
                        # Extract properties if node or relationship
                        processed_record = {}
                        for key, value in record.items():
                            if hasattr(value, 'items'):  # Node or relationship
                                processed_record[key] = dict(value.items())
                            elif hasattr(value, 'start_node') and hasattr(value, 'type'):  # Relationship
                                processed_record[key] = f"{value.type} ({value.start_node.element_id} -> {value.end_node.element_id})"
                            elif hasattr(value, 'nodes') and hasattr(value, 'relationships'):  # Path
                                # Create a more meaningful representation of the path
                                nodes_info = []
                                for node in value.nodes:
                                    # Get node label and a display name (either name or title property)
                                    node_label = next(iter(node.labels), "")
                                    node_props = dict(node.items())
                                    display_name = node_props.get('name', node_props.get('title', f"Node({node.element_id})"))
                                    nodes_info.append(f"{node_label}:'{display_name}'")
                                
                                # Get relationship types
                                rel_types = [rel.type for rel in value.relationships]
                                
                                # Format as: "Path: Movie:'The Matrix' -[ACTED_IN]-> Person:'Keanu Reeves'"
                                path_str = "Path: " + nodes_info[0]
                                for i, rel_type in enumerate(rel_types):
                                    path_str += f" -[{rel_type}]-> {nodes_info[i+1]}"
                                
                                processed_record[key] = path_str
                            elif value is None:
                                processed_record[key] = None
                            else:
                                processed_record[key] = value
                        data.append(processed_record)
                    
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df)
                    else:
                        st.info("Query executed successfully, but returned no tabular data")
                else:  # Graph visualization
                    # Check if result contains nodes and relationships
                    has_graph_data = False
                    
                    for record in result:
                        for key, value in record.items():
                            if hasattr(value, 'nodes') and hasattr(value, 'relationships'):  # Path
                                has_graph_data = True
                                break
                            elif hasattr(value, 'labels') or hasattr(value, 'type'):  # Node or Relationship
                                has_graph_data = True
                                break
                    
                    if has_graph_data:
                        # Create NetworkX graph from Neo4j result
                        G = nx.DiGraph()
                        
                        # Keep track of nodes we've seen
                        nodes_seen = {}
                        
                        for record in result:
                            for key, value in record.items():
                                # Handle paths
                                if hasattr(value, 'nodes') and hasattr(value, 'relationships'):
                                    for node in value.nodes:
                                        if node.element_id not in nodes_seen:
                                            node_props = dict(node.items())
                                            node_label = next(iter(node.labels), "")
                                            display_name = node_props.get('name', node_props.get('title', str(node.element_id)))
                                            
                                            # Remove name if it exists in props to avoid conflict
                                            if 'name' in node_props:
                                                del node_props['name']
                                                
                                            G.add_node(node.element_id, label=node_label, name=display_name, **node_props)
                                            nodes_seen[node.element_id] = True
                                    
                                    for rel in value.relationships:
                                        start_id = rel.start_node.element_id
                                        end_id = rel.end_node.element_id
                                        rel_props = dict(rel.items())
                                        G.add_edge(start_id, end_id, type=rel.type, **rel_props)
                                
                                # Handle individual nodes
                                elif hasattr(value, 'labels'):
                                    if value.element_id not in nodes_seen:
                                        node_props = dict(value.items())
                                        node_label = next(iter(value.labels), "")
                                        display_name = node_props.get('name', node_props.get('title', str(value.element_id)))
                                        
                                        # Remove name if it exists in props to avoid conflict
                                        if 'name' in node_props:
                                            del node_props['name']
                                            
                                        G.add_node(value.element_id, label=node_label, name=display_name, **node_props)
                                        nodes_seen[value.element_id] = True
                                
                                # Handle individual relationships
                                elif hasattr(value, 'start_node') and hasattr(value, 'end_node'):
                                    start_id = value.start_node.element_id
                                    end_id = value.end_node.element_id
                                    rel_props = dict(value.items())
                                    
                                    # Add nodes if not seen yet
                                    if start_id not in nodes_seen:
                                        start_props = dict(value.start_node.items())
                                        start_label = next(iter(value.start_node.labels), "")
                                        start_name = start_props.get('name', start_props.get('title', str(start_id)))
                                        
                                        # Remove name if it exists in props to avoid conflict
                                        if 'name' in start_props:
                                            del start_props['name']
                                            
                                        G.add_node(start_id, label=start_label, name=start_name, **start_props)
                                        nodes_seen[start_id] = True
                                        
                                    if end_id not in nodes_seen:
                                        end_props = dict(value.end_node.items())
                                        end_label = next(iter(value.end_node.labels), "")
                                        end_name = end_props.get('name', end_props.get('title', str(end_id)))
                                        
                                        # Remove name if it exists in props to avoid conflict
                                        if 'name' in end_props:
                                            del end_props['name']
                                            
                                        G.add_node(end_id, label=end_label, name=end_name, **end_props)
                                        nodes_seen[end_id] = True
                                    
                                    G.add_edge(start_id, end_id, type=value.type, **rel_props)
                        
                        if G.number_of_nodes() > 0:
                            # Create PyVis network
                            net = Network(notebook=True, cdn_resources="remote", height="600px", width="100%")
                            
                            # Configure physics for better visualizations of crowded graphs
                            # Increase repulsion between nodes and spring length for more spacing
                            net.barnes_hut(
                                gravity=-repulsion_strength,      # Strong negative gravity to push nodes apart
                                central_gravity=0.2,              # Reduced central gravity for less clustering
                                spring_length=graph_spacing,      # Controls spacing between connected nodes
                                spring_strength=spring_strength / 1000.0  # Convert percentage to small decimal
                            )
                            
                            # Add nodes with properties
                            for node_id, node_data in G.nodes(data=True):
                                label = node_data.get('name', str(node_id))
                                title = "<br>".join([f"{k}: {v}" for k, v in node_data.items() 
                                                    if k not in ['name', 'id'] and v is not None])
                                
                                # Color by label
                                node_type = node_data.get('label', 'Unknown')
                                if 'Movie' in node_type:
                                    color = '#ff9999'  # Red for movies
                                elif 'Person' in node_type:
                                    color = '#99ccff'  # Blue for people
                                elif 'Genre' in node_type:
                                    color = '#99ff99'  # Green for genres
                                else:
                                    color = '#f0f0f0'  # Light gray for others
                                
                                # Use element_id instead of id (to fix deprecation warning)
                                net.add_node(node_id, 
                                            label=label, 
                                            title=title, 
                                            color=color, 
                                            physics=True,
                                            size=25,  # Larger node size 
                                            font={'size': 14, 'color': '#000000', 'face': 'arial', 'strokeWidth': 2, 'strokeColor': '#ffffff'}  # Improved font with white outline
                                )
                            
                            # Add edges with types
                            for source, target, edge_data in G.edges(data=True):
                                rel_type = edge_data.get('type', '')
                                title = f"Type: {rel_type}<br>" + "<br>".join([f"{k}: {v}" for k, v in edge_data.items() 
                                                                  if k != 'type' and v is not None])
                                
                                weight = edge_data.get('weight', 1)
                                # If weight is present, make edge width proportional
                                width = 1 + min(5, weight) if 'weight' in edge_data else 1
                                
                                net.add_edge(source, target, 
                                           title=title, 
                                           label=rel_type, 
                                           width=width,
                                           arrows={'to': {'enabled': True, 'scaleFactor': 0.8}},  # Nice arrows
                                           font={'size': 12, 'color': '#000000', 'strokeWidth': 2, 'strokeColor': '#ffffff'},  # Improved font with outline
                                           length=edge_length  # Use user-defined edge length
                                )
                            
                            # Generate the HTML visualization
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
                                net.save_graph(tmpfile.name)
                                with open(tmpfile.name, 'r', encoding='utf-8') as f:
                                    html_data = f.read()
                                st.components.v1.html(html_data, height=600)
                        else:
                            st.warning("Query returned no graph data to visualize")
                    else:
                        st.error("Query results don't contain graph elements. For graph visualization, return paths or nodes.")
                        st.info("Example query for graph: MATCH path = (m:Movie)-[:ACTED_IN]-(p:Person) WHERE m.title CONTAINS 'Matrix' RETURN path LIMIT 20")
                        
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

else:
    st.info("Please connect to Neo4j using the sidebar to begin.") 