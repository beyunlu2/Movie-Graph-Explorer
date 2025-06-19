import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
from graph_utils import Neo4jConnector
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from node2vec import Node2Vec

class MovieKnowledgeGraph:
    """
    Class to design and populate a knowledge graph for movie data in Neo4j.
    """
    
    def __init__(self, connector):
        """
        Initialize with a Neo4j connector.
        
        Args:
            connector: Neo4jConnector instance
        """
        self.connector = connector
    
    def create_constraints(self):
        """
        Create constraints for node uniqueness in Neo4j.
        """
        # Create constraints for Movie nodes
        self.connector.execute_query("""
        CREATE CONSTRAINT movie_id IF NOT EXISTS 
        FOR (m:Movie) REQUIRE m.id IS UNIQUE
        """)
        
        # Create constraints for Person nodes
        self.connector.execute_query("""
        CREATE CONSTRAINT person_id IF NOT EXISTS 
        FOR (p:Person) REQUIRE p.id IS UNIQUE
        """)
        
        # Create constraints for Genre nodes
        self.connector.execute_query("""
        CREATE CONSTRAINT genre_name IF NOT EXISTS 
        FOR (g:Genre) REQUIRE g.name IS UNIQUE
        """)
        
        # Create constraints for Country nodes
        self.connector.execute_query("""
        CREATE CONSTRAINT country_name IF NOT EXISTS 
        FOR (c:Country) REQUIRE c.name IS UNIQUE
        """)
        
        # Create constraints for Company nodes
        self.connector.execute_query("""
        CREATE CONSTRAINT company_id IF NOT EXISTS 
        FOR (c:Company) REQUIRE c.id IS UNIQUE
        """)
        
        print("Constraints created successfully")
    
    def create_indexes(self):
        """
        Create indexes for better query performance.
        """
        # Create index on Movie title
        self.connector.execute_query("""
        CREATE INDEX movie_title IF NOT EXISTS
        FOR (m:Movie) ON (m.title)
        """)
        
        # Create index on Person name
        self.connector.execute_query("""
        CREATE INDEX person_name IF NOT EXISTS
        FOR (p:Person) ON (p.name)
        """)
        
        print("Indexes created successfully")
    
    def create_similar_movie_relationships(self):
        """
        Create SIMILAR_TO relationships between movies based on shared genres and cast.
        """
        self.connector.execute_query("""
        MATCH (m1:Movie)-[:IN_GENRE]->(g:Genre)<-[:IN_GENRE]-(m2:Movie)
        WHERE m1.id < m2.id
        WITH m1, m2, count(g) AS shared_genres
        WHERE shared_genres >= 2
        
        MATCH (m1)<-[:ACTED_IN]-(a:Person)-[:ACTED_IN]->(m2)
        WITH m1, m2, shared_genres, count(a) AS shared_actors
        WHERE shared_actors > 0
        
        MERGE (m1)-[r:SIMILAR_TO]-(m2)
        SET r.score = shared_genres * 3 + shared_actors * 2
        """)
        
        print("Created SIMILAR_TO relationships between movies")
    
    def create_recommended_relationships(self):
        """
        Create RECOMMENDED relationships for movies with similar genres but different eras.
        """
        self.connector.execute_query("""
        MATCH (m1:Movie)-[:IN_GENRE]->(g:Genre)<-[:IN_GENRE]-(m2:Movie)
        WHERE abs(m1.release_year - m2.release_year) > 20
        AND m1.vote_average > 7.0
        AND m2.vote_average > 7.0
        WITH m1, m2, collect(g.name) AS shared_genres, count(g) AS genre_count
        WHERE genre_count >= 2
        
        MERGE (m1)-[r:RECOMMENDS]->(m2)
        SET r.shared_genres = shared_genres,
            r.score = genre_count * m2.vote_average / 10
        """)
        
        print("Created RECOMMENDS relationships between movies from different eras")

    def create_knowledge_graph(self, data_directory=None):
        """
        Set up the knowledge graph by creating constraints, indexes, and derived relationships.
        Assumes the movie dataset already exists in Neo4j.
        """
        print("Setting up knowledge graph (no data import, using existing Neo4j data)...")
        self.create_constraints()
        self.create_indexes()
        self.create_similar_movie_relationships()
        self.create_recommended_relationships()
        print("Knowledge graph setup completed (no data import performed)")

    def get_graph_statistics(self):
        """
        Get statistics about the knowledge graph using standard Cypher queries.
        
        Returns:
            dict: Statistics about nodes and relationships
        """
        try:
            # Count nodes by label using standard Cypher
            node_query = """
            CALL db.labels() YIELD label
            MATCH (n:`${label}`)
            RETURN label AS nodeType, count(n) AS nodeCount
            ORDER BY nodeCount DESC
            """
            
            # We need to run this for each label since Neo4j doesn't have a built-in way
            # to count by all labels in one query without APOC
            labels_result = self.connector.execute_query("CALL db.labels() YIELD label")
            node_counts = []
            
            if labels_result:
                for record in labels_result:
                    label = record["label"]
                    count_query = f"MATCH (n:`{label}`) RETURN '{label}' AS nodeType, count(n) AS nodeCount"
                    result = self.connector.execute_query(count_query)
                    if result and len(result) > 0:
                        node_counts.append(result[0])
                
            # Count relationships by type using standard Cypher
            rel_query = """
            CALL db.relationshipTypes() YIELD relationshipType
            MATCH ()-[r:`${relationshipType}`]->()
            RETURN relationshipType AS relType, count(r) AS count
            ORDER BY count DESC
            """
            
            # Same for relationship types
            rel_types_result = self.connector.execute_query("CALL db.relationshipTypes() YIELD relationshipType")
            rel_counts = []
            
            if rel_types_result:
                for record in rel_types_result:
                    rel_type = record["relationshipType"]
                    count_query = f"MATCH ()-[r:`{rel_type}`]->() RETURN '{rel_type}' AS relType, count(r) AS count"
                    result = self.connector.execute_query(count_query)
                    if result and len(result) > 0:
                        rel_counts.append(result[0])
            
            # Compile statistics
            stats = {
                "nodes": {record["nodeType"]: record["nodeCount"] for record in node_counts},
                "relationships": {record["relType"]: record["count"] for record in rel_counts}
            }
            
            return stats
        except Exception as e:
            print(f"Error getting graph statistics: {e}")
            # Return empty stats rather than None to avoid 'NoneType' errors
            return {"nodes": {}, "relationships": {}}


class MovieGraphAnalysis:
    """
    Class for performing exploratory analysis and subgraph analysis on the movie knowledge graph.
    """
    
    def __init__(self, connector):
        """
        Initialize with a Neo4j connector.
        
        Args:
            connector: Neo4jConnector instance
        """
        self.connector = connector
        
    def run_cypher_query(self, query, params=None):
        """
        Run a Cypher query and return results as a pandas DataFrame.
        
        Args:
            query: Cypher query string
            params: Query parameters (optional)
            
        Returns:
            pandas DataFrame with query results
        """
        results = self.connector.execute_query(query, params or {})
        
        if not results:
            return pd.DataFrame()
        
        # Convert Neo4j Records to dictionary format
        data = []
        for record in results:
            data.append({key: record[key] for key in record.keys()})
        
        return pd.DataFrame(data)
    
    def get_node_labels(self):
        """
        Get all node labels from the database.
        
        Returns:
            List of node labels
        """
        query = """
        CALL db.labels() YIELD label
        RETURN label
        ORDER BY label
        """
        df = self.run_cypher_query(query)
        
        if df.empty:
            return []
        
        return df['label'].tolist()
    
    def get_relationship_types(self):
        """
        Get all relationship types from the database.
        
        Returns:
            List of relationship types
        """
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN relationshipType
        ORDER BY relationshipType
        """
        df = self.run_cypher_query(query)
        
        if df.empty:
            return []
        
        return df['relationshipType'].tolist()
    
    def get_relationship_types_for_node(self, node_label):
        """
        Get relationship types that are connected to a specific node label.
        
        Args:
            node_label: Node label to get relationships for
            
        Returns:
            List of relationship types
        """
        query = f"""
        MATCH (n:{node_label})-[r]->()
        RETURN DISTINCT type(r) AS relationshipType
        UNION
        MATCH (n:{node_label})<-[r]-()
        RETURN DISTINCT type(r) AS relationshipType
        ORDER BY relationshipType
        """
        df = self.run_cypher_query(query)
        
        if df.empty:
            return []
        
        return df['relationshipType'].tolist()
    
    def get_graph_statistics(self):
        """
        Get basic statistics about the graph.
        
        Returns:
            Dictionary with node and relationship counts
        """
        # Count nodes by label
        node_query = """
        MATCH (n)
        RETURN labels(n)[0] AS nodeType, count(n) AS count
        ORDER BY count DESC
        """
        node_counts = self.run_cypher_query(node_query)
        
        # Count relationships by type
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS relType, count(r) AS count
        ORDER BY count DESC
        """
        rel_counts = self.run_cypher_query(rel_query)
        
        # Print summary
        print("Graph Statistics:")
        print(f"Total nodes: {node_counts['count'].sum() if not node_counts.empty else 0}")
        print(f"Total relationships: {rel_counts['count'].sum() if not rel_counts.empty else 0}")
        
        # Convert DataFrames to dictionaries for easier access in the app
        node_counts_dict = {}
        if not node_counts.empty:
            for _, row in node_counts.iterrows():
                node_counts_dict[row['nodeType']] = row['count']
                
        rel_counts_dict = {}
        if not rel_counts.empty:
            for _, row in rel_counts.iterrows():
                rel_counts_dict[row['relType']] = row['count']
        
        # Return dictionary format for app.py
        return {
            "node_counts": node_counts_dict,
            "relationship_counts": rel_counts_dict,
            "nodes": node_counts_dict,  # For backward compatibility
            "relationships": rel_counts_dict  # For backward compatibility
        }
    
    def plot_graph_statistics(self, node_counts, rel_counts):
        """
        Plot node and relationship counts.
        
        Args:
            node_counts: DataFrame with node counts
            rel_counts: DataFrame with relationship counts
        """
        plt.figure(figsize=(14, 10))
        
        # Plot node counts
        plt.subplot(2, 1, 1)
        sns.barplot(x='nodeType', y='count', data=node_counts)
        plt.title('Node Count by Type')
        plt.ylabel('Count')
        plt.xlabel('Node Type')
        plt.xticks(rotation=45)
        
        # Plot relationship counts
        plt.subplot(2, 1, 2)
        sns.barplot(x='relType', y='count', data=rel_counts)
        plt.title('Relationship Count by Type')
        plt.ylabel('Count')
        plt.xlabel('Relationship Type')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('graph_statistics.png')
        plt.show()
    
    def analyze_degree_distribution(self, node_label=None):
        """
        Analyze the degree distribution of nodes in the graph.
        
        Args:
            node_label: Optionally filter by node label
            
        Returns:
            DataFrame with degree distribution
        """
        # Build query based on whether node_label is provided
        if node_label:
            query = f"""
            MATCH (n:{node_label})
            RETURN id(n) AS nodeId, 
                   n.name AS name,
                   apoc.node.degree(n, 'OUT') AS outDegree,
                   apoc.node.degree(n, 'IN') AS inDegree,
                   apoc.node.degree(n) AS totalDegree
            ORDER BY totalDegree DESC
            LIMIT 1000
            """
        else:
            query = """
            MATCH (n)
            RETURN id(n) AS nodeId, 
                   labels(n)[0] AS nodeType,
                   apoc.node.degree(n, 'OUT') AS outDegree,
                   apoc.node.degree(n, 'IN') AS inDegree,
                   apoc.node.degree(n) AS totalDegree
            ORDER BY totalDegree DESC
            LIMIT 1000
            """
        
        degree_df = self.run_cypher_query(query)
        
        if degree_df.empty:
            print("No degree data found")
            return None
        
        print(f"Degree Statistics:")
        print(f"Max degree: {degree_df['totalDegree'].max()}")
        print(f"Min degree: {degree_df['totalDegree'].min()}")
        print(f"Mean degree: {degree_df['totalDegree'].mean():.2f}")
        print(f"Median degree: {degree_df['totalDegree'].median()}")
        
        return degree_df
    
    def plot_degree_distribution(self, degree_df, node_label=None):
        """
        Plot the degree distribution.
        
        Args:
            degree_df: DataFrame with degree data
            node_label: Node label for title (optional)
        """
        plt.figure(figsize=(15, 10))
        
        # Plot degree histogram
        plt.subplot(2, 2, 1)
        sns.histplot(degree_df['totalDegree'], bins=30, kde=True)
        plt.title(f'Degree Distribution {node_label or "All Nodes"}')
        plt.xlabel('Degree')
        plt.ylabel('Count')
        plt.yscale('log')
        
        # Plot out-degree histogram
        plt.subplot(2, 2, 2)
        sns.histplot(degree_df['outDegree'], bins=30, kde=True)
        plt.title(f'Out-Degree Distribution {node_label or "All Nodes"}')
        plt.xlabel('Out-Degree')
        plt.ylabel('Count')
        plt.yscale('log')
        
        # Plot in-degree histogram
        plt.subplot(2, 2, 3)
        sns.histplot(degree_df['inDegree'], bins=30, kde=True)
        plt.title(f'In-Degree Distribution {node_label or "All Nodes"}')
        plt.xlabel('In-Degree')
        plt.ylabel('Count')
        plt.yscale('log')
        
        # Plot scatter of in vs out degree
        plt.subplot(2, 2, 4)
        sns.scatterplot(x='inDegree', y='outDegree', data=degree_df, alpha=0.6)
        plt.title(f'In-Degree vs Out-Degree {node_label or "All Nodes"}')
        plt.xlabel('In-Degree')
        plt.ylabel('Out-Degree')
        
        plt.tight_layout()
        plt.savefig(f'degree_distribution_{node_label or "all"}.png')
        plt.show()
        
    def analyze_centrality(self, node_type, limit=100, algorithm='pagerank'):
        """
        Analyze node centrality in the graph using various algorithms.
        
        Args:
            node_type: Type of node to analyze (e.g., Person, Movie)
            limit: Number of top nodes to return
            algorithm: Centrality algorithm - 'pagerank', 'betweenness', 'closeness', or 'degree'
            
        Returns:
            DataFrame with centrality scores
        """
        # Different algorithms available in Neo4j Graph Data Science (GDS) library
        if algorithm.lower() == 'pagerank':
            query = f"""
            CALL gds.pageRank.stream('movieGraph')
            YIELD nodeId, score
            MATCH (n) WHERE id(n) = nodeId AND n:{node_type}
            RETURN n.name AS name, labels(n)[0] AS type, score AS centrality
            ORDER BY centrality DESC
            LIMIT {limit}
            """
        elif algorithm.lower() == 'betweenness':
            query = f"""
            CALL gds.betweenness.stream('movieGraph')
            YIELD nodeId, score
            MATCH (n) WHERE id(n) = nodeId AND n:{node_type}
            RETURN n.name AS name, labels(n)[0] AS type, score AS centrality
            ORDER BY centrality DESC
            LIMIT {limit}
            """
        elif algorithm.lower() == 'closeness':
            query = f"""
            CALL gds.closeness.stream('movieGraph')
            YIELD nodeId, score
            MATCH (n) WHERE id(n) = nodeId AND n:{node_type}
            RETURN n.name AS name, labels(n)[0] AS type, score AS centrality
            ORDER BY centrality DESC
            LIMIT {limit}
            """
        elif algorithm.lower() == 'degree':
            query = f"""
            MATCH (n:{node_type})
            RETURN n.name AS name, 
                   labels(n)[0] AS type, 
                   apoc.node.degree(n) AS centrality
            ORDER BY centrality DESC
            LIMIT {limit}
            """
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        try:
            centrality_df = self.run_cypher_query(query)
            
            if centrality_df.empty:
                print(f"No centrality data found for {node_type} using {algorithm}")
                return None
            
            print(f"Top {min(limit, len(centrality_df))} {node_type} nodes by {algorithm} centrality:")
            print(centrality_df.head(10))
            
            return centrality_df
        except Exception as e:
            print(f"Error running centrality analysis: {e}")
            print("If using GDS algorithms, make sure the graph projection exists:")
            print("CALL gds.graph.project('movieGraph', ['Movie', 'Person', 'Genre'], '*')")
            return None
    
    def plot_centrality(self, centrality_df, algorithm, node_type):
        """
        Plot centrality results.
        
        Args:
            centrality_df: DataFrame with centrality data
            algorithm: Centrality algorithm used
            node_type: Type of node analyzed
        """
        plt.figure(figsize=(12, 6))
        
        # Plot the top nodes by centrality
        top_n = min(20, len(centrality_df))
        plt.subplot(1, 2, 1)
        sns.barplot(x='centrality', y='name', data=centrality_df.head(top_n))
        plt.title(f'Top {top_n} {node_type} by {algorithm} Centrality')
        plt.xlabel('Centrality Score')
        plt.ylabel(node_type)
        
        # Plot the centrality distribution
        plt.subplot(1, 2, 2)
        sns.histplot(centrality_df['centrality'], kde=True)
        plt.title(f'{algorithm} Centrality Distribution for {node_type}')
        plt.xlabel('Centrality')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f'{algorithm}_centrality_{node_type}.png')
        plt.show()
        
    def run_community_detection(self, algorithm='louvain', limit=1000):
        """
        Perform community detection on the graph.
        
        Args:
            algorithm: Community detection algorithm - 'louvain', 'label_propagation', or 'strongly_connected'
            limit: Maximum number of nodes to process
            
        Returns:
            DataFrame with community assignments
        """
        if algorithm.lower() == 'louvain':
            query = f"""
            CALL gds.louvain.stream('movieGraph')
            YIELD nodeId, communityId
            MATCH (n) WHERE id(n) = nodeId
            RETURN n.name AS name, labels(n)[0] AS type, communityId
            ORDER BY communityId, type, name
            LIMIT {limit}
            """
        elif algorithm.lower() == 'label_propagation':
            query = f"""
            CALL gds.labelPropagation.stream('movieGraph')
            YIELD nodeId, communityId
            MATCH (n) WHERE id(n) = nodeId
            RETURN n.name AS name, labels(n)[0] AS type, communityId
            ORDER BY communityId, type, name
            LIMIT {limit}
            """
        elif algorithm.lower() == 'strongly_connected':
            query = f"""
            CALL gds.alpha.scc.stream('movieGraph')
            YIELD nodeId, componentId
            MATCH (n) WHERE id(n) = nodeId
            RETURN n.name AS name, labels(n)[0] AS type, componentId AS communityId
            ORDER BY communityId, type, name
            LIMIT {limit}
            """
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        try:
            community_df = self.run_cypher_query(query)
            
            if community_df.empty:
                print(f"No community data found using {algorithm}")
                return None
            
            # Count nodes in each community
            community_counts = community_df.groupby('communityId').size().reset_index(name='count')
            community_counts = community_counts.sort_values('count', ascending=False)
            
            print(f"Found {len(community_counts)} communities using {algorithm}")
            print(f"Top communities by size:")
            print(community_counts.head(10))
            
            return community_df, community_counts
        except Exception as e:
            print(f"Error running community detection: {e}")
            print("If using GDS algorithms, make sure the graph projection exists:")
            print("CALL gds.graph.project('movieGraph', ['Movie', 'Person', 'Genre'], '*')")
            return None
    
    def analyze_community(self, community_df, community_id):
        """
        Analyze the composition of a specific community.
        
        Args:
            community_df: DataFrame with community assignments
            community_id: ID of the community to analyze
            
        Returns:
            DataFrame with community composition
        """
        # Filter for the specific community
        community = community_df[community_df['communityId'] == community_id]
        
        if community.empty:
            print(f"No data found for community {community_id}")
            return None
        
        # Count node types in the community
        type_counts = community.groupby('type').size().reset_index(name='count')
        type_counts = type_counts.sort_values('count', ascending=False)
        
        print(f"Community {community_id} Composition:")
        print(f"Total nodes: {len(community)}")
        print("Node types:")
        print(type_counts)
        
        # Get sample nodes from each type
        samples = []
        for node_type in type_counts['type']:
            type_nodes = community[community['type'] == node_type]
            samples.append(type_nodes.head(5))
            
        sample_df = pd.concat(samples)
        print("\nSample nodes from each type:")
        print(sample_df)
        
        return community, type_counts
    
    def plot_community_analysis(self, community_counts, community_df=None):
        """
        Plot community detection results.
        
        Args:
            community_counts: DataFrame with community sizes
            community_df: Full DataFrame with community assignments (optional)
        """
        plt.figure(figsize=(15, 10))
        
        # Plot community size distribution
        plt.subplot(2, 2, 1)
        sns.histplot(community_counts['count'], bins=30, kde=True)
        plt.title('Community Size Distribution')
        plt.xlabel('Community Size')
        plt.ylabel('Frequency')
        plt.yscale('log')
        
        # Plot top communities by size
        plt.subplot(2, 2, 2)
        top_n = min(20, len(community_counts))
        sns.barplot(x='count', y='communityId', data=community_counts.head(top_n))
        plt.title(f'Top {top_n} Communities by Size')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Community ID')
        
        if community_df is not None:
            # Plot node type distribution across top communities
            plt.subplot(2, 1, 2)
            top_communities = community_counts.head(5)['communityId'].tolist()
            top_comm_data = community_df[community_df['communityId'].isin(top_communities)]
            
            # Count each type in each community
            pivot_data = pd.crosstab(
                index=top_comm_data['communityId'], 
                columns=top_comm_data['type'],
                normalize='index'
            )
            
            # Plot stacked bars
            pivot_data.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.title('Node Type Distribution in Top 5 Communities')
            plt.xlabel('Community ID')
            plt.ylabel('Proportion')
            plt.legend(title='Node Type')
        
        plt.tight_layout()
        plt.savefig('community_analysis.png')
        plt.show()
    
    def analyze_subgraph(self, cypher_query, params=None, name="subgraph"):
        """
        Extract and analyze a subgraph using a Cypher query.
        
        Args:
            cypher_query: Cypher query to extract the subgraph
            params: Query parameters (optional)
            name: Name for the subgraph (for saving files)
            
        Returns:
            NetworkX graph object
        """
        results = self.connector.execute_query(cypher_query, params or {})
        
        if not results:
            print("No results found for subgraph query")
            return None
        
        # Create a NetworkX graph
        G = nx.DiGraph()
        
        # Process records to extract nodes and edges
        nodes = {}
        for record in results:
            # Process all items in the record
            for key, value in record.items():
                # Handle individual nodes
                if hasattr(value, 'labels') and hasattr(value, 'id'):
                    node_id = value.element_id
                    if node_id not in nodes:
                        # Extract properties
                        props = dict(value.items())
                        node_label = next(iter(value.labels), "Unknown")
                        
                        # Add node to the graph
                        G.add_node(node_id, 
                                  label=node_label,
                                  name=props.get('name', str(node_id)),
                                  **props)
                        nodes[node_id] = True
                        
                # Check if this is a relationship
                elif hasattr(value, 'start_node') and hasattr(value, 'end_node'):
                    start_id = value.start_node.element_id
                    end_id = value.end_node.element_id
                    rel_type = value.type
                    
                    # Add the relationship to the graph
                    props = dict(value.items())
                    G.add_edge(start_id, end_id, 
                              type=rel_type,
                              **props)
                    
                    # Make sure the nodes exist
                    if start_id not in nodes:
                        start_node = value.start_node
                        start_props = dict(start_node.items())
                        start_label = next(iter(start_node.labels), "Unknown")
                        G.add_node(start_id, 
                                  label=start_label,
                                  name=start_props.get('name', str(start_id)),
                                  **start_props)
                        nodes[start_id] = True
                        
                    if end_id not in nodes:
                        end_node = value.end_node
                        end_props = dict(end_node.items())
                        end_label = next(iter(end_node.labels), "Unknown")
                        G.add_node(end_id, 
                                  label=end_label,
                                  name=end_props.get('name', str(end_id)),
                                  **end_props)
                        nodes[end_id] = True
        
        print(f"Extracted subgraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Basic analysis
        if G.number_of_nodes() > 0:
            # Count node types
            node_types = [data['label'] for _, data in G.nodes(data=True)]
            type_counts = Counter(node_types)
            print("\nNode types in subgraph:")
            for node_type, count in type_counts.most_common():
                print(f"  {node_type}: {count}")
            
            # Count edge types
            edge_types = [data['type'] for _, _, data in G.edges(data=True)]
            edge_type_counts = Counter(edge_types)
            print("\nRelationship types in subgraph:")
            for edge_type, count in edge_type_counts.most_common():
                print(f"  {edge_type}: {count}")
            
            # Calculate basic metrics
            if G.number_of_nodes() > 1:
                density = nx.density(G)
                print(f"\nGraph density: {density:.6f}")
                
                # Check if graph is connected (ignoring direction)
                undirected = G.to_undirected()
                is_connected = nx.is_connected(undirected)
                print(f"Is connected: {is_connected}")
                
                if not is_connected:
                    components = list(nx.connected_components(undirected))
                    print(f"Number of connected components: {len(components)}")
                    print(f"Largest component size: {len(max(components, key=len))}")
        
        return G
    
    def visualize_subgraph(self, G, name="subgraph", max_nodes=100):
        """
        Visualize a subgraph using NetworkX and Matplotlib.
        
        Args:
            G: NetworkX graph object
            name: Name for the subgraph (for saving files)
            max_nodes: Maximum number of nodes to visualize
        """
        if G is None or G.number_of_nodes() == 0:
            print("No graph to visualize")
            return
        
        if G.number_of_nodes() > max_nodes:
            print(f"Graph too large to visualize ({G.number_of_nodes()} nodes). Showing largest component...")
            # Extract largest connected component
            undirected = G.to_undirected()
            largest_cc = max(nx.connected_components(undirected), key=len)
            G = G.subgraph(largest_cc).copy()
            
            # Further reduce if still too large
            if G.number_of_nodes() > max_nodes:
                print(f"Largest component still too large ({G.number_of_nodes()} nodes). Showing top {max_nodes} nodes by degree...")
                # Keep top N nodes by degree
                degrees = dict(G.degree())
                top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
                G = G.subgraph(top_nodes).copy()
        
        plt.figure(figsize=(15, 12))
        
        # Get node labels and types for coloring
        node_labels = {node: data.get('name', str(node)) for node, data in G.nodes(data=True)}
        node_types = [data.get('label', 'Unknown') for _, data in G.nodes(data=True)]
        
        # Get unique node types for color mapping
        unique_types = list(set(node_types))
        color_map = plt.cm.get_cmap('tab10', len(unique_types))
        
        # Assign colors based on node type
        colors = [color_map(unique_types.index(t)) for t in node_types]
        
        # Position nodes using a layout algorithm
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the graph
        plt.subplot(1, 1, 1)
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=250, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)
        
        # Draw node labels for smaller graphs
        if G.number_of_nodes() <= 50:
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
        
        # Create a legend for node types
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map(i), markersize=10, label=t) 
                  for i, t in enumerate(unique_types)]
        plt.legend(handles=handles, title='Node Types', loc='upper right')
        
        plt.title(f'Subgraph Visualization: {name} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'subgraph_{name}.png', dpi=300)
        plt.show()

    def project_cooccurrence_network(self, node_label='Person', rel_type='ACTED_IN', projected_graph_name='cooccurrenceGraph'):
        """
        Project a co-occurrence (monopartite) network in Neo4j using GDS.
        E.g., actor-actor network based on co-acting in movies.
        """
        # First check if a projection already exists with that name and delete it
        try:
            check_query = f"""
            CALL gds.graph.exists('{projected_graph_name}')
            YIELD exists
            RETURN exists
            """
            
            check_result = self.run_cypher_query(check_query)
            if not check_result.empty and check_result.iloc[0]['exists']:
                drop_query = f"""
                CALL gds.graph.drop('{projected_graph_name}')
                YIELD graphName
                RETURN graphName
                """
                self.connector.execute_query(drop_query)
                print(f"Dropped existing projection: {projected_graph_name}")
        except Exception as e:
            print(f"Error checking/dropping projection: {e}")
        
        # Determine if we need to create a bipartite or monopartite projection
        # For bipartite projection (e.g., Person-ACTED_IN->Movie), we need a different approach
        is_bipartite = False
        target_label = None
        
        # First check if this is a valid relationship type for this node
        rel_exists_query = f"""
        MATCH (:{node_label})-[r:{rel_type}]-()
        RETURN count(r) > 0 AS exists
        """
        
        rel_exists_result = self.run_cypher_query(rel_exists_query)
        if rel_exists_result.empty or not rel_exists_result.iloc[0]['exists']:
            error_msg = f"No {rel_type} relationships found for {node_label} nodes"
            print(error_msg)
            return 0, 0
        
        # Check what's on the other side of the relationship
        target_node_query = f"""
        MATCH (:{node_label})-[r:{rel_type}]->(target)
        RETURN labels(target)[0] AS targetLabel, count(*) AS count
        LIMIT 1
        """
        
        target_result = self.run_cypher_query(target_node_query)
        if not target_result.empty:
            target_label = target_result.iloc[0]['targetLabel']
            if target_label != node_label:
                is_bipartite = True
                print(f"Detected bipartite relationship: {node_label}-{rel_type}->{target_label}")
        else:
            # Try reverse direction
            reverse_query = f"""
            MATCH (target)-[r:{rel_type}]->(:{node_label})
            RETURN labels(target)[0] AS targetLabel, count(*) AS count
            LIMIT 1
            """
            
            reverse_result = self.run_cypher_query(reverse_query)
            if not reverse_result.empty:
                target_label = reverse_result.iloc[0]['targetLabel']
                if target_label != node_label:
                    is_bipartite = True
                    print(f"Detected bipartite relationship in reverse direction: {target_label}-{rel_type}->{node_label}")
        
        # For bipartite relationships (like Person-ACTED_IN->Movie), we need to project a bipartite graph first
        # then use projection methods to create relationships between people who acted in the same movie
        if is_bipartite and target_label:
            try:
                # Create the bipartite projection
                bipartite_name = f"{projected_graph_name}"
                
                # Try to drop existing bipartite projection
                try:
                    self.connector.execute_query(f"CALL gds.graph.drop('{bipartite_name}')")
                except:
                    pass
                
                # Create the bipartite projection
                bipartite_query = f"""
                CALL gds.graph.project(
                    '{bipartite_name}',
                    ['{node_label}', '{target_label}'],
                    {{
                        {rel_type}: {{
                            type: '{rel_type}',
                            orientation: 'NATURAL'
                        }}
                    }}
                )
                """
                
                self.connector.execute_query(bipartite_query)
                print(f"Created bipartite projection: {bipartite_name}")
                
                # Now project a monopartite graph using cypher projection
                # Handle both directions of the relationship
                projection_query = f"""
                CALL gds.graph.project.cypher(
                    '{projected_graph_name}',
                    'MATCH (n:{node_label}) RETURN id(n) AS id, n.name AS name',
                    'MATCH (a:{node_label})-[:{rel_type}]->(b)<-[:{rel_type}]-(c:{node_label})
                     WHERE id(a) < id(c)
                     RETURN id(a) AS source, id(c) AS target, count(*) AS weight'
                )
                YIELD graphName, nodeCount, relationshipCount
                RETURN graphName, nodeCount, relationshipCount
                """
                
                result = self.connector.execute_query(projection_query)
                if result:
                    node_count = result[0]["nodeCount"]
                    rel_count = result[0]["relationshipCount"]
                    print(f"Projected {projected_graph_name} as a {node_label} co-occurrence network with {node_count} nodes and {rel_count} relationships.")
                    
                    if rel_count == 0:
                        # If no relationships were created, try alternative Cypher
                        try:
                            self.connector.execute_query(f"CALL gds.graph.drop('{projected_graph_name}')")
                        except:
                            pass
                            
                        alt_query = f"""
                        CALL gds.graph.project.cypher(
                            '{projected_graph_name}',
                            'MATCH (n:{node_label}) RETURN id(n) AS id, n.name AS name',
                            'MATCH (a:{node_label})-[:{rel_type}]-(b)-[:{rel_type}]-(c:{node_label})
                             WHERE id(a) < id(c)
                             RETURN id(a) AS source, id(c) AS target, count(*) AS weight'
                        )
                        """
                        
                        alt_result = self.connector.execute_query(alt_query)
                        if alt_result:
                            print(f"Re-projected {projected_graph_name} using alternative query.")
                            return 1, 1  # Return success 
                    

                    
                    return node_count, rel_count
                else:
                    print("Failed to create co-occurrence projection.")
                    # Fallback to creating direct relationships in the database
                    print("Creating direct co-occurrence relationships as fallback...")
                    rel_name = f"CO_OCCURS_WITH_{node_label.upper()}"
                    self.create_cooccurrence_network(
                        node_label=node_label,
                        rel_type=rel_type,
                        output_rel_name=rel_name
                    )
                    return 0, 0
                    
            except Exception as e:
                print(f"Error projecting co-occurrence network: {e}")
                # Fallback to creating direct relationships in the database
                print("Creating direct co-occurrence relationships as fallback...")
                rel_name = f"CO_OCCURS_WITH_{node_label.upper()}"
                self.create_cooccurrence_network(
                    node_label=node_label,
                    rel_type=rel_type,
                    output_rel_name=rel_name
                )
                return 0, 0
        else:
            # For monopartite relationships (like Person-KNOWS->Person), project directly
            query = f'''
            CALL gds.graph.project(
                '{projected_graph_name}',
                '{node_label}',
                {{
                    {rel_type}: {{
                        orientation: 'UNDIRECTED',
                        properties: {{
                            weight: {{
                                property: 'weight',
                                defaultValue: 1.0
                            }}
                        }}
                    }}
                }}
            )
            '''
            
            try:
                result = self.connector.execute_query(query)
                if result:
                    print(f"Projected {projected_graph_name} as a {node_label} monopartite network.")
                    return 1, 1  # Placeholder values, we don't have actual counts
                else:
                    print("Failed to create monopartite projection.")
                    return 0, 0
            except Exception as e:
                print(f"Error projecting monopartite network: {e}")
                
                # Fallback to creating direct relationships in the database
                print("Creating direct co-occurrence relationships as fallback...")
                rel_name = f"CO_OCCURS_WITH_{node_label.upper()}"
                self.create_cooccurrence_network(
                    node_label=node_label,
                    rel_type=rel_type,
                    output_rel_name=rel_name
                )
                return 0, 0

    def run_node_similarity(self, projected_graph_name='cooccurrenceGraph', top_k=10, bottom_k=0, algorithm='jaccard'):
        """
        Run GDS nodeSimilarity to construct a similarity graph (e.g., based on shared movies).
        Returns a DataFrame of similar node pairs and their similarity scores.
        
        Args:
            projected_graph_name: Name of the GDS graph projection
            top_k: Number of top results to return (limit) or 0 for no limit
            bottom_k: Number of bottom results to return (limit) or 0 for no limit
            algorithm: Similarity algorithm to use - 'jaccard', 'cosine', or 'overlap'
            
        Returns:
            DataFrame with node pairs and similarity scores
        """
        
        # First check if we have a valid projection
        if not self.is_gds_available():
            print("GDS is not available. Using custom implementation instead.")
            return self.calculate_node_similarity(
                node_label="Person", 
                co_occurs_rel="CO_OCCURS_WITH", 
                top_k=top_k,
                bottom_k=bottom_k,
                algorithm=algorithm
            )
        
        # Try to run with GDS
        # Try different GDS procedure names and yield parameters
        procedures = [
            # Format: [procedure_name, yield_fields]
            ["gds.nodeSimilarity.stream", "node1, node2, similarity"],
            ["gds.similarity.node.stream", "node1, node2, similarity"],  # Newer naming convention
            ["gds.nodeSimilarity.filtered.stream", "node1, node2, similarity"],
        ]
        
        df = None
        errors = []
        
        # Check if projection exists first
        try:
            check_query = f"""
            CALL gds.graph.exists('{projected_graph_name}')
            YIELD exists
            RETURN exists
            """
            
            check_result = self.run_cypher_query(check_query)
            if check_result.empty or not check_result.iloc[0]['exists']:
                print(f"Warning: Projection '{projected_graph_name}' does not exist. Attempting to use custom implementation.")
                # Create co-occurrence relationships as a fallback
                self.create_cooccurrence_network(
                    node_label="Person", 
                    rel_type="ACTED_IN", 
                    output_rel_name="CO_OCCURS_WITH"
                )
                return self.calculate_node_similarity(
                    node_label="Person", 
                    co_occurs_rel="CO_OCCURS_WITH", 
                    top_k=top_k,
                    bottom_k=bottom_k,
                    algorithm=algorithm
                )
        except Exception as e:
            print(f"Warning: Error checking if projection exists: {str(e)}")
            # Continue trying different procedures
        
        # Try different procedure names
        for proc, yields in procedures:
            # Build the query with the specified algorithm and limits
            # Set appropriate limit parameters
            if top_k > 0:
                limit_param = f", {{topK: {top_k}}}"
            elif bottom_k > 0:
                limit_param = f", {{bottomK: {bottom_k}}}"
            else:
                limit_param = ""
            
            # Build the configuration for the algorithm
            if algorithm.lower() == 'cosine':
                config = f", {{similarityMetric: 'cosine'{', topK: ' + str(top_k) if top_k > 0 else ''}{', bottomK: ' + str(bottom_k) if bottom_k > 0 else ''}}}"
            elif algorithm.lower() == 'overlap':
                config = f", {{similarityMetric: 'overlap'{', topK: ' + str(top_k) if top_k > 0 else ''}{', bottomK: ' + str(bottom_k) if bottom_k > 0 else ''}}}"
            else:  # Default to Jaccard
                config = limit_param  # Jaccard is the default, just add limit parameters
            
            # Add LIMIT clause directly in the query for when using no config limits
            limit_sql = f"LIMIT {top_k}" if top_k > 0 else (f"LIMIT {bottom_k}" if bottom_k > 0 else "")
            
            query = f'''
            CALL {proc}('{projected_graph_name}'{config})
            YIELD {yields}
            WITH node1, node2, similarity 
            WHERE gds.util.asNode(node1).name < gds.util.asNode(node2).name
            RETURN gds.util.asNode(node1).name AS node1, gds.util.asNode(node2).name AS node2, similarity
            ORDER BY similarity {('DESC' if top_k > 0 else 'ASC')}
            {limit_sql}
            '''
            
            try:
                df = self.run_cypher_query(query)
                if not df.empty:
                    print(f"Successfully used {proc} for node similarity with {algorithm} algorithm")
                    # Rename the similarity column to include the algorithm used
                    df = df.rename(columns={'similarity': f'similarity({algorithm.capitalize()})'})
                    return df
            except Exception as e:
                errors.append(f"Error with {proc}: {str(e)}")
                continue
        
        # All GDS procedures failed, try our custom implementation with co-occurrence network
        print("All GDS node similarity procedures failed. Using custom implementation.")
        # Only log the first few errors to avoid excessive output
        if len(errors) > 3:
            print(f"First 3 of {len(errors)} errors:")
            print("\n".join(errors[:3]))
        else:
            print("\n".join(errors))
        
        # First check if we have co-occurrence relationships
        count_query = """
        MATCH ()-[r:CO_OCCURS_WITH]->()
        RETURN count(r) AS rel_count
        """
        count_result = self.run_cypher_query(count_query)
        rel_count = count_result.iloc[0]['rel_count'] if not count_result.empty else 0
        
        # If no co-occurrence relationships exist, create them first
        if rel_count == 0:
            print("No CO_OCCURS_WITH relationships found. Creating co-occurrence network first...")
            result = self.create_cooccurrence_network(
                node_label="Person", 
                rel_type="ACTED_IN", 
                output_rel_name="CO_OCCURS_WITH"
            )
            print(result)
        
        # Now try to calculate similarity
        try:
            df = self.calculate_node_similarity(
                node_label="Person", 
                co_occurs_rel="CO_OCCURS_WITH", 
                top_k=top_k,
                bottom_k=bottom_k,
                algorithm=algorithm
            )
            return df
        except Exception as e:
            print(f"Error running custom node similarity: {e}")
            return pd.DataFrame()

    def run_jaccard_link_prediction(self, projected_graph_name='cooccurrenceGraph', top_k=10):
        """
        Perform link prediction using Jaccard Similarity in Neo4j GDS.
        Returns a DataFrame of predicted links and their scores.
        
        Args:
            projected_graph_name: Name of the GDS graph projection
            top_k: Number of top results to return (limit) or 0 for no limit
            
        Returns:
            DataFrame with node pairs and similarity scores
        """
            
        # First check if we have a valid projection
        if not self.is_gds_available():
            print("GDS is not available. Using custom implementation instead.")
            return self.calculate_link_prediction(
                node_label="Person", 
                co_occurs_rel="CO_OCCURS_WITH", 
                method="jaccard", 
                top_k=top_k
            )
        
        # Check if projection exists first
        try:
            check_query = f"""
            CALL gds.graph.exists('{projected_graph_name}')
            YIELD exists
            RETURN exists
            """
            
            check_result = self.run_cypher_query(check_query)
            if check_result.empty or not check_result.iloc[0]['exists']:
                print(f"Warning: Projection '{projected_graph_name}' does not exist. Attempting to use custom implementation.")
                # Create co-occurrence relationships as a fallback
                self.create_cooccurrence_network(
                    node_label="Person", 
                    rel_type="ACTED_IN", 
                    output_rel_name="CO_OCCURS_WITH"
                )
                return self.calculate_link_prediction(
                    node_label="Person", 
                    co_occurs_rel="CO_OCCURS_WITH", 
                    method="jaccard", 
                    top_k=top_k
                )
        except Exception as e:
            print(f"Warning: Error checking if projection exists: {str(e)}")
            # Continue trying different procedures
        
        # Try different GDS procedure names based on version
        procedures = [
            "gds.nodeSimilarity.stream",  # Fallback to node similarity
            "gds.jaccard.stream",  # Direct jaccard
            "gds.similarity.jaccard.stream",  # Newer naming convention
            "gds.linkPrediction.jaccard.stream",  # Newer version
            "gds.alpha.linkprediction.jaccard.stream",  # Alpha version 
            "gds.beta.linkprediction.jaccard.stream",  # Beta version
            "gds.beta.similarity.jaccard.stream"  # Another possible name
        ]
        
        df = None
        errors = []
        
        # Try different procedure names
        for procedure in procedures:
            # Handle different yield parameters based on procedure name
            if "nodeSimilarity" in procedure:
                yields = "node1, node2, similarity"
                score_field = "similarity"
            else:
                yields = "node1, node2, similarity"
                score_field = "similarity"
            
            # Add LIMIT clause only if top_k > 0
            limit_sql = f"LIMIT {top_k}" if top_k > 0 else ""
            
            query = f'''
            CALL {procedure}('{projected_graph_name}')
            YIELD {yields}
            WITH node1, node2, {score_field} AS score
            WHERE gds.util.asNode(node1).name < gds.util.asNode(node2).name
            RETURN gds.util.asNode(node1).name AS node1, gds.util.asNode(node2).name AS node2, score
            ORDER BY score DESC
            {limit_sql}
            '''
            
            try:
                df = self.run_cypher_query(query)
                if not df.empty:
                    print(f"Successfully used {procedure} for link prediction")
                    return df
            except Exception as e:
                errors.append(f"Error with {procedure}: {str(e)}")
                continue
        
        # All GDS procedures failed, try our custom implementation with co-occurrence network
        print("All GDS procedures failed. Using custom implementation.")
        # Only log the first few errors to avoid excessive output
        if len(errors) > 3:
            print(f"First 3 of {len(errors)} errors:")
            print("\n".join(errors[:3]))
        else:
            print("\n".join(errors))
        
        # First check if we have co-occurrence relationships
        count_query = """
        MATCH ()-[r:CO_OCCURS_WITH]->()
        RETURN count(r) AS rel_count
        """
        count_result = self.run_cypher_query(count_query)
        rel_count = count_result.iloc[0]['rel_count'] if not count_result.empty else 0
        
        # If no co-occurrence relationships exist, create them first
        if rel_count == 0:
            print("No CO_OCCURS_WITH relationships found. Creating co-occurrence network first...")
            result = self.create_cooccurrence_network(
                node_label="Person", 
                rel_type="ACTED_IN", 
                output_rel_name="CO_OCCURS_WITH"
            )
            print(result)
        
        # Use our non-GDS implementation instead
        try:
            df = self.calculate_link_prediction(
                node_label="Person", 
                co_occurs_rel="CO_OCCURS_WITH", 
                method="jaccard", 
                top_k=top_k
            )
            return df
        except Exception as e:
            print(f"Error running custom link prediction: {e}")
            return pd.DataFrame()

    def run_link_prediction(self, projected_graph_name='cooccurrenceGraph', top_k=10):
        """
        Perform link prediction using GDS (e.g., using Adamic-Adar).
        Returns a DataFrame of predicted links and their scores.
        
        Args:
            projected_graph_name: Name of the GDS graph projection
            top_k: Number of top results to return (limit) or 0 for no limit
            
        Returns:
            DataFrame with node pairs and similarity scores
        """
        # First check if we have a valid projection
        if not self.is_gds_available():
            print("GDS is not available. Using custom implementation instead.")
            return self.calculate_link_prediction(
                node_label="Person", 
                co_occurs_rel="CO_OCCURS_WITH", 
                method="adamic_adar", 
                top_k=top_k
            )
        
        # Check if projection exists first
        try:
            check_query = f"""
            CALL gds.graph.exists('{projected_graph_name}')
            YIELD exists
            RETURN exists
            """
            
            check_result = self.run_cypher_query(check_query)
            if check_result.empty or not check_result.iloc[0]['exists']:
                print(f"Warning: Projection '{projected_graph_name}' does not exist. Attempting to use custom implementation.")
                # Create co-occurrence relationships as a fallback
                self.create_cooccurrence_network(
                    node_label="Person", 
                    rel_type="ACTED_IN", 
                    output_rel_name="CO_OCCURS_WITH"
                )
                return self.calculate_link_prediction(
                    node_label="Person", 
                    co_occurs_rel="CO_OCCURS_WITH", 
                    method="adamic_adar", 
                    top_k=top_k
                )
        except Exception as e:
            print(f"Warning: Error checking if projection exists: {str(e)}")
            # Continue trying different procedures
            
        # Try different GDS procedure names and yield parameters
        procedures = [
            # Format: [procedure_name, yield_fields, score_field]
            ["gds.nodeSimilarity.stream", "node1, node2, similarity", "similarity"],  # Fallback to similarity
            ["gds.adamicAdar.stream", "node1, node2, score", "score"],  # Direct adamic adar
            ["gds.similarity.adamicAdar.stream", "node1, node2, score", "score"],  # Newer naming convention
            ["gds.linkPrediction.adamicAdar.stream", "node1, node2, score", "score"],  # Newer version
        ]
        
        df = None
        errors = []
        
        # Try different procedure names
        for proc, yields, score_field in procedures:
            # Add LIMIT clause only if top_k > 0
            limit_sql = f"LIMIT {top_k}" if top_k > 0 else ""
            
            query = f'''
            CALL {proc}('{projected_graph_name}')
            YIELD {yields}
            WITH node1, node2, {score_field} AS score
            WHERE gds.util.asNode(node1).name < gds.util.asNode(node2).name
            RETURN gds.util.asNode(node1).name AS node1, gds.util.asNode(node2).name AS node2, score
            ORDER BY score DESC
            {limit_sql}
            '''
            
            try:
                df = self.run_cypher_query(query)
                if not df.empty:
                    print(f"Successfully used {proc} for link prediction")
                    return df
            except Exception as e:
                errors.append(f"Error with {proc}: {str(e)}")
                continue
        
        # All GDS procedures failed, try our custom implementation with co-occurrence network
        print("All GDS link prediction procedures failed. Using custom implementation.")
        # Only log the first few errors to avoid excessive output
        if len(errors) > 3:
            print(f"First 3 of {len(errors)} errors:")
            print("\n".join(errors[:3]))
        else:
            print("\n".join(errors))
        
        # First check if we have co-occurrence relationships
        count_query = """
        MATCH ()-[r:CO_OCCURS_WITH]->()
        RETURN count(r) AS rel_count
        """
        count_result = self.run_cypher_query(count_query)
        rel_count = count_result.iloc[0]['rel_count'] if not count_result.empty else 0
        
        # If no co-occurrence relationships exist, create them first
        if rel_count == 0:
            print("No CO_OCCURS_WITH relationships found. Creating co-occurrence network first...")
            result = self.create_cooccurrence_network(
                node_label="Person", 
                rel_type="ACTED_IN", 
                output_rel_name="CO_OCCURS_WITH"
            )
            print(result)
        
        # Use our non-GDS implementation instead
        try:
            df = self.calculate_link_prediction(
                node_label="Person", 
                co_occurs_rel="CO_OCCURS_WITH", 
                method="adamic_adar", 
                top_k=top_k
            )
            return df
        except Exception as e:
            print(f"Error running custom link prediction: {e}")
            return pd.DataFrame()

    def run_knowledge_graph_completion(self, projected_graph_name='cooccurrenceGraph', top_k=10, method='pykeen_transe'):
        """
        Perform knowledge graph completion to suggest new relationships in the graph.
        This implementation offers multiple approaches including:
        1. PyKEEN embeddings (TransE, RotatE, PairRE) - State-of-the-art KG embeddings
        2. Link prediction based on graph topology
        3. Rule-based completion using patterns in the data
        
        Args:
            projected_graph_name: Name of the GDS graph projection (for non-PyKEEN methods)
            top_k: Number of top results to return
            method: Completion method ('pykeen_transe', 'pykeen_rotate', 'pykeen_pairre', 
                                      'link_prediction', 'rule')
            
        Returns:
            DataFrame with suggested new relationships
        """
        print(f"Running knowledge graph completion using {method} method...")
        
        # Import PyKEEN KG completion module if using those methods
        if method.startswith('pykeen_'):
            try:
                from python_kg_completion import PykeenMovieKG
                
                # Initialize PyKEEN with our connector
                pykeen_kg = PykeenMovieKG(self.connector)
                
                # Determine model type
                model_name = method.replace('pykeen_', '').capitalize()
                if model_name == 'Transe':
                    model_name = 'TransE'
                elif model_name == 'Rotate':
                    model_name = 'RotatE'
                elif model_name == 'Pairre':
                    model_name = 'PairRE'
                
                # Try to load existing model
                try:
                    print(f"Loading existing {model_name} model...")
                    pykeen_kg.train_model(model_name=model_name, load_model=True)
                except Exception as e:
                    print(f"Could not load model: {e}. Will use pre-extracted predictions.")
                
                # Run batch link prediction to find new relationships
                if pykeen_kg.trained:
                    try:
                        # Get predictions for new relationships
                        predictions = pykeen_kg.batch_link_prediction(k=top_k)
                        
                        if not predictions.empty:
                            # Format for standard interface
                            result_df = pd.DataFrame({
                                'node1': predictions['head'],
                                'node2': predictions['tail'],
                                'rel_type': predictions['relation'],
                                'score': predictions['confidence'],
                                'method': f"PyKEEN_{model_name}"
                            })
                            return result_df.sort_values('score', ascending=False).head(top_k)
                    except Exception as e:
                        print(f"Error running PyKEEN predictions: {e}")
                
                # Fallback to pre-extracted predictions if model prediction fails
                print("Using pre-extracted semantic relationships based on knowledge graph patterns...")
                query = """
                MATCH (m1:Movie)-[:IN_GENRE]->(g:Genre)<-[:IN_GENRE]-(m2:Movie)
                WHERE m1 <> m2
                WITH m1, m2, COUNT(g) AS genreCount
                WHERE genreCount > 1
                MATCH (m1)<-[:ACTED_IN]-(a:Person)
                MATCH (m2)<-[:ACTED_IN]-(a)
                WITH m1, m2, COUNT(a) AS actorOverlap, genreCount
                WHERE actorOverlap > 0
                RETURN 
                    m1.title AS node1, 
                    m2.title AS node2, 
                    'SIMILAR_TO' AS rel_type,
                    (actorOverlap * 0.6 + genreCount * 0.4)/10 AS score,
                    'PyKEEN_fallback' AS method
                ORDER BY score DESC
                LIMIT $top_k
                """
                try:
                    return self.run_cypher_query(query, {"top_k": top_k})
                except Exception as e:
                    print(f"Error running fallback query: {e}")
                    # If all else fails, use link prediction
                    return self.run_link_prediction(projected_graph_name, top_k)
            
            except ImportError as e:
                print(f"Error importing PyKEEN module: {e}. Falling back to link prediction.")
                return self.run_link_prediction(projected_graph_name, top_k)
        
        elif method == 'link_prediction':
            # Use link prediction as a basic approach to knowledge graph completion
            print("Using topology-based link prediction for knowledge graph completion...")
            return self.run_link_prediction(projected_graph_name, top_k)
        
        elif method == 'rule':
            # Rule-based completion uses patterns in data to infer new relationships
            print("Using rule-based approach for knowledge graph completion...")
            
            # Enhanced rule: If A and B co-acted in multiple films, and B and C also co-acted,
            # and A directed films that have similar genres to C's films, A might collaborate with C
            rule_query = """
            MATCH (a:Person)-[:ACTED_IN]->(:Movie)<-[:ACTED_IN]-(b:Person),
                  (b:Person)-[:ACTED_IN]->(:Movie)<-[:ACTED_IN]-(c:Person)
            WHERE a <> c AND a <> b AND b <> c
            AND NOT (a)-[:ACTED_IN]->(:Movie)<-[:ACTED_IN]-(c)
            
            WITH a, c, count(b) AS connection_strength
            
            // Check for genre compatibility
            OPTIONAL MATCH (a)-[:DIRECTED|ACTED_IN]->(m1:Movie)-[:IN_GENRE]->(g:Genre)<-[:IN_GENRE]-(m2:Movie)<-[:ACTED_IN|DIRECTED]-(c)
            WITH a, c, connection_strength, count(DISTINCT g) AS genre_overlap
            
            RETURN a.name AS node1, c.name AS node2, 
                   'POTENTIAL_COLLABORATION' AS rel_type,
                   connection_strength * 0.5 + genre_overlap * 0.7 AS score,
                   'Rule_based' AS method
            ORDER BY score DESC
            LIMIT $top_k
            """
            
            try:
                return self.run_cypher_query(rule_query, {"top_k": top_k})
            except Exception as e:
                print(f"Error running rule-based completion: {e}")
                print("Falling back to link prediction...")
                return self.run_link_prediction(projected_graph_name, top_k)
        
        # For backward compatibility, default to embedding-based approach
        elif method in ['transe', 'embedding']:
            print("Using embedding-based approach for knowledge graph completion...")
            
            # Use Node2Vec embeddings (can be replaced with TransE later)
            try:
                # Use a more robust approach to build the knowledge subgraph
                subgraph_query = """
                MATCH (p:Person)
                WHERE (p)-[:ACTED_IN|DIRECTED]->(:Movie)
                WITH p LIMIT 300
                
                MATCH (p)-[r:ACTED_IN|DIRECTED]->(m:Movie)
                RETURN id(p) AS source, id(m) AS target, type(r) AS type
                LIMIT 5000
                """
                
                # Include names to make the results more interpretable
                name_query = """
                MATCH (n) 
                WHERE id(n) IN $ids
                RETURN id(n) AS id, 
                       CASE WHEN n:Person THEN n.name ELSE n.title END AS name,
                       CASE WHEN n:Person THEN 'Person' ELSE 'Movie' END AS label
                """
                
                edges_df = self.run_cypher_query(subgraph_query)
                
                if edges_df.empty:
                    print("No data available for embedding. Falling back to link prediction.")
                    return self.run_link_prediction(projected_graph_name, top_k)
                
                # Build knowledge graph
                G = nx.DiGraph()
                
                # Extract unique node IDs
                all_nodes = set(edges_df['source'].tolist() + edges_df['target'].tolist())
                
                # Get node names
                nodes_df = self.run_cypher_query(name_query, {"ids": list(all_nodes)})
                node_info = {row['id']: {'name': row['name'], 'label': row['label']} 
                            for _, row in nodes_df.iterrows()}
                
                # Add nodes with attributes
                for node_id, info in node_info.items():
                    G.add_node(node_id, name=info['name'], label=info['label'])
                
                # Add edges with types
                for _, row in edges_df.iterrows():
                    G.add_edge(row['source'], row['target'], type=row['type'])
                
                print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                
                # Use Node2Vec to generate embeddings and predict links
                try:
                    from node2vec import Node2Vec
                    import numpy as np
                    from sklearn.metrics.pairwise import cosine_similarity
                    
                    # Generate embeddings
                    node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=50, workers=1)
                    model = node2vec.fit(window=5, min_count=1, batch_words=4)
                    
                    # Separate nodes by type
                    person_nodes = [n for n, d in G.nodes(data=True) if d.get('label') == 'Person']
                    movie_nodes = [n for n, d in G.nodes(data=True) if d.get('label') == 'Movie']
                    
                    # For each person, find potential new movie connections
                    potential_relations = []
                    
                    # Consider only nodes that have embeddings
                    embedded_person_nodes = [p for p in person_nodes if str(p) in model.wv]
                    embedded_movie_nodes = [m for m in movie_nodes if str(m) in model.wv]
                    
                    # Get embeddings
                    person_embeddings = {p: model.wv[str(p)] for p in embedded_person_nodes}
                    movie_embeddings = {m: model.wv[str(m)] for m in embedded_movie_nodes}
                    
                    # For potential actor-movie relations
                    for p in embedded_person_nodes[:100]:  # Limit for performance
                        p_embedding = person_embeddings[p]
                        p_name = G.nodes[p].get('name', 'Unknown')
                        
                        # Find movies this person has not worked with
                        for m in embedded_movie_nodes[:200]:  # Limit for performance
                            if G.has_edge(p, m):
                                continue  # Skip existing connections
                                
                            m_embedding = movie_embeddings[m]
                            m_name = G.nodes[m].get('name', 'Unknown')
                            
                            # Calculate similarity
                            similarity = np.dot(p_embedding, m_embedding) / (
                                np.linalg.norm(p_embedding) * np.linalg.norm(m_embedding))
                            
                            potential_relations.append({
                                'node1': p_name,
                                'node2': m_name,
                                'node1_type': 'Person',
                                'node2_type': 'Movie',
                                'score': similarity,
                                'rel_type': 'POTENTIAL_RELATION',
                                'method': 'Embedding'
                            })
                    
                    # For potential actor-actor collaborations
                    for i, p1 in enumerate(embedded_person_nodes[:50]):  # Limit for performance
                        p1_embedding = person_embeddings[p1]
                        p1_name = G.nodes[p1].get('name', 'Unknown')
                        
                        for p2 in embedded_person_nodes[i+1:50]:  # Avoid duplicates
                            # Skip if they've already collaborated
                            has_collaboration = False
                            for m in movie_nodes:
                                if G.has_edge(p1, m) and G.has_edge(p2, m):
                                    has_collaboration = True
                                    break
                            
                            if has_collaboration:
                                continue
                                
                            p2_embedding = person_embeddings[p2]
                            p2_name = G.nodes[p2].get('name', 'Unknown')
                            
                            # Calculate similarity
                            similarity = np.dot(p1_embedding, p2_embedding) / (
                                np.linalg.norm(p1_embedding) * np.linalg.norm(p2_embedding))
                            
                            potential_relations.append({
                                'node1': p1_name,
                                'node2': p2_name,
                                'node1_type': 'Person',
                                'node2_type': 'Person',
                                'score': similarity,
                                'rel_type': 'POTENTIAL_COLLABORATION',
                                'method': 'Embedding'
                            })
                    
                    # Create result DataFrame and sort by score
                    result_df = pd.DataFrame(potential_relations)
                    result_df = result_df.sort_values('score', ascending=False).head(top_k)
                    return result_df
                    
                except Exception as e:
                    print(f"Error in embedding-based completion: {e}")
                    return self.run_link_prediction(projected_graph_name, top_k)
                
            except Exception as e:
                print(f"Error creating subgraph: {e}")
                return self.run_link_prediction(projected_graph_name, top_k)
        
        else:
            # Fallback to link prediction for unknown methods
            print(f"Unknown method '{method}'. Falling back to link prediction.")
            return self.run_link_prediction(projected_graph_name, top_k)

    def evaluate_link_prediction(self, y_true, y_scores, threshold=0.5):
        """
        Evaluate link prediction results using precision, recall, F1, and AUC.
        y_true: array-like of true labels (1 for real link, 0 for non-link)
        y_scores: array-like of predicted scores (from link prediction algorithm)
        threshold: score threshold for classifying as link
        Returns: dict of metrics
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        import numpy as np
        y_pred = (np.array(y_scores) >= threshold).astype(int)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_scores)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

    def run_ml_link_prediction(self, nx_graph, algorithm='logistic_regression', test_frac=0.3, random_state=42):
        """
        Perform link prediction using Node2Vec embeddings and a selected ML classifier.
        
        Args:
            nx_graph: networkx.Graph (undirected, e.g., actor-actor or movie-movie)
            algorithm: ML algorithm to use ('logistic_regression', 'random_forest', or 'support_vector_machine')
            test_frac: fraction of edges to use for testing
            random_state: random seed for reproducibility
            
        Returns: 
            y_true, y_scores (for evaluation), and dataframe with predicted links
        """

        # Prepare positive and negative edge samples
        edges = list(nx_graph.edges())
        non_edges = list(nx.non_edges(nx_graph))
        np.random.seed(random_state)
        neg_sample = np.random.choice(len(non_edges), size=len(edges), replace=False)
        neg_edges = [non_edges[i] for i in neg_sample]

        # Split into train/test
        pos_train, pos_test = train_test_split(edges, test_size=test_frac, random_state=random_state)
        neg_train, neg_test = train_test_split(neg_edges, test_size=test_frac, random_state=random_state)

        # Remove test edges from graph for embedding
        G_train = nx_graph.copy()
        G_train.remove_edges_from(pos_test)
        G_train.remove_edges_from(neg_test)  # (neg edges are not in graph, but safe)

        # Node2Vec embedding
        node2vec = Node2Vec(G_train, dimensions=64, walk_length=10, num_walks=50, workers=1, seed=random_state)
        model = node2vec.fit(window=5, min_count=1, batch_words=4)

        def edge_embedding(u, v):
            # Check if both nodes exist in the embedding vocabulary
            if str(u) not in model.wv or str(v) not in model.wv:
                # Return zero vector with same dimensions as the embeddings
                return np.zeros(model.wv.vector_size)
            return model.wv[str(u)] * model.wv[str(v)]  # Hadamard product, ensure nodes are strings
        
        # Filter edges to only include those with both nodes in the embedding
        valid_pos_train = [(u, v) for u, v in pos_train if str(u) in model.wv and str(v) in model.wv]
        valid_neg_train = [(u, v) for u, v in neg_train if str(u) in model.wv and str(v) in model.wv]
        valid_pos_test = [(u, v) for u, v in pos_test if str(u) in model.wv and str(v) in model.wv]
        valid_neg_test = [(u, v) for u, v in neg_test if str(u) in model.wv and str(v) in model.wv]
        
        # If we have no valid edges for training, return empty results
        if not valid_pos_train or not valid_neg_train:
            print("Not enough valid edges for training. Try with a more connected graph.")
            return [], [], pd.DataFrame()
            
        # If we have no valid edges for testing, return empty results
        if not valid_pos_test or not valid_neg_test:
            print("Not enough valid edges for testing. Try with a more connected graph.")
            return [], [], pd.DataFrame()

        # Prepare train data with valid edges only
        X_train = [edge_embedding(u, v) for u, v in valid_pos_train + valid_neg_train]
        y_train = [1] * len(valid_pos_train) + [0] * len(valid_neg_train)
        
        # Prepare test data with valid edges only
        X_test = [edge_embedding(u, v) for u, v in valid_pos_test + valid_neg_test]
        y_test = [1] * len(valid_pos_test) + [0] * len(valid_neg_test)
        test_edges = valid_pos_test + valid_neg_test

        # Train classifier based on selected algorithm
        if algorithm == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, random_state=random_state)
        elif algorithm == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif algorithm == 'support_vector_machine':
            from sklearn.svm import SVC
            clf = SVC(probability=True, random_state=random_state)
        else:
            # Default to logistic regression if algorithm not recognized
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, random_state=random_state)
            
        # Train model and get predictions
        clf.fit(X_train, y_train)
        y_scores = clf.predict_proba(X_test)[:, 1]
        
        # Create a DataFrame with predictions for visualization
        edge_data = []
        for i, ((node1, node2), score) in enumerate(zip(test_edges, y_scores)):
            # Get node names from IDs if available
            node1_name = nx_graph.nodes[node1].get('name', str(node1)) if 'name' in nx_graph.nodes[node1] else str(node1)
            node2_name = nx_graph.nodes[node2].get('name', str(node2)) if 'name' in nx_graph.nodes[node2] else str(node2)
            
            is_true_link = 1 if i < len(valid_pos_test) else 0
            edge_data.append({
                'node1': node1_name,
                'node2': node2_name,
                'score': score,
                'is_true_link': is_true_link
            })
        
        # Create DataFrame and sort by prediction score
        predictions_df = pd.DataFrame(edge_data)
        predictions_df = predictions_df.sort_values('score', ascending=False)
        
        return y_test, y_scores, predictions_df

    def is_gds_available(self):
        """
        Check if Neo4j Graph Data Science library is available.
        Returns: bool
        """
        # Try multiple different queries to check for GDS availability
        check_queries = [
            # Basic check for newer GDS versions
            "CALL gds.list() YIELD name RETURN count(*) > 0 AS available LIMIT 1",
            # Alternative check for older GDS versions
            "CALL gds.listProcs() YIELD name RETURN count(*) > 0 AS available LIMIT 1",
            # Simple check for GDS 1.x
            "CALL gds.version() YIELD version RETURN true AS available",
            # Check for alpha procedures in older versions
            "CALL gds.alpha.list() YIELD name RETURN count(*) > 0 AS available LIMIT 1", 
            # Another alternative for older versions
            "CALL dbms.procedures() YIELD name WHERE name STARTS WITH 'gds.' RETURN count(*) > 0 AS available LIMIT 1"
        ]
        
        for query in check_queries:
            try:
                result = self.connector.execute_query(query)
                if result and result[0].get("available", False):
                    print(f"GDS is available (detected via '{query}')")
                    return True
            except Exception:
                # Continue to the next check
                continue
        
        print("GDS is not available after trying multiple detection methods")
        return False
            
    def create_cooccurrence_network(self, node_label, rel_type, output_rel_name='CO_OCCURS_WITH'):
        """
        Create a co-occurrence network directly in Neo4j without using GDS.
        This is an alternative to the projection method when GDS is not available.
        
        Args:
            node_label: Node type to analyze (e.g., Person, Movie)
            rel_type: Relationship type connecting the nodes (e.g., ACTED_IN)
            output_rel_name: Name for the new co-occurrence relationship
        """
        # First check for existing relationships to avoid duplicates
        check_query = f"""
        MATCH ()-[r:`{output_rel_name}`]->()
        RETURN count(r) AS count
        """
        
        check_result = self.connector.execute_query(check_query)
        existing_count = check_result[0]["count"] if check_result else 0
        
        if existing_count > 0:
            print(f"Found {existing_count} existing {output_rel_name} relationships. Clearing them first...")
            clear_query = f"""
            MATCH ()-[r:`{output_rel_name}`]->()
            DELETE r
            """
            self.connector.execute_query(clear_query)
        
        # Check if source nodes and relationships exist
        source_check = f"""
        MATCH (n:`{node_label}`)
        RETURN count(n) AS node_count
        """
        
        rel_check = f"""
        MATCH ()-[r:`{rel_type}`]->()
        RETURN count(r) AS rel_count
        """
        
        source_result = self.connector.execute_query(source_check)
        rel_result = self.connector.execute_query(rel_check)
        
        node_count = source_result[0]["node_count"] if source_result else 0
        rel_count = rel_result[0]["rel_count"] if rel_result else 0
        
        print(f"Found {node_count} {node_label} nodes and {rel_count} {rel_type} relationships")
        
        if node_count == 0 or rel_count == 0:
            return f"Error: Not enough data to create co-occurrence network. Need {node_label} nodes and {rel_type} relationships."
        
        # Handle bipartite networks like Person-ACTED_IN->Movie
        # Create direct relationships between Persons who acted in the same Movie
        print(f"Creating co-occurrence network for {node_label} nodes connected via {rel_type}...")
        
        # More efficient query that directly calculates weight
        query = f"""
        MATCH (a:`{node_label}`)-[:`{rel_type}`]->(b)<-[:`{rel_type}`]-(c:`{node_label}`)
        WHERE id(a) < id(c)  // Avoid duplicate relationships
        WITH a, c, count(b) AS weight
        WHERE weight > 0
        CREATE (a)-[r:`{output_rel_name}`]->(c)
        SET r.weight = weight
        RETURN count(*) AS created_relationships
        """
        
        # Alternative query if the first one doesn't work with direction
        alt_query = f"""
        MATCH (a:`{node_label}`)-[:`{rel_type}`]-(b)-[:`{rel_type}`]-(c:`{node_label}`)
        WHERE id(a) < id(c)  // Avoid duplicate relationships
        WITH a, c, count(DISTINCT b) AS weight
        WHERE weight > 0
        CREATE (a)-[r:`{output_rel_name}`]->(c)
        SET r.weight = weight
        RETURN count(*) AS created_relationships
        """
        
        try:
            result = self.connector.execute_query(query)
            count = result[0]["created_relationships"] if result else 0
            
            if count == 0:
                print("First query didn't create any relationships. Trying alternative query...")
                result = self.connector.execute_query(alt_query)
                count = result[0]["created_relationships"] if result else 0
            
            if count > 0:
                print(f"Successfully created {count} {output_rel_name} relationships between {node_label} nodes")
                
                # Add a simple query to show example relationships
                sample_query = f"""
                MATCH (a:`{node_label}`)-[r:`{output_rel_name}`]-(b:`{node_label}`)
                RETURN a.name AS node1, b.name AS node2, r.weight AS weight
                ORDER BY r.weight DESC
                LIMIT 5
                """
                
                sample_result = self.connector.execute_query(sample_query)
                if sample_result:
                    print("Sample relationships:")
                    for record in sample_result:
                        print(f"  {record['node1']} -- {record['weight']} --> {record['node2']}")
                
                return f"Created {count} {output_rel_name} relationships between {node_label} nodes"
            else:
                return f"No {output_rel_name} relationships created. Please check your data model."
                
        except Exception as e:
            error_msg = f"Error creating co-occurrence network: {str(e)}"
            print(error_msg)
            return error_msg
            
    def calculate_node_similarity(self, node_label, co_occurs_rel='CO_OCCURS_WITH', top_k=10, bottom_k=0, algorithm='jaccard'):
        """
        Find similar nodes based on co-occurrence patterns without using GDS.
        This is an alternative when GDS nodeSimilarity is not available.
        
        Args:
            node_label: Node type to analyze
            co_occurs_rel: Co-occurrence relationship type
            top_k: Number of top results to return (0 for unlimited)
            bottom_k: Number of bottom results to return (0 for unlimited)
            algorithm: Similarity algorithm to use ('jaccard', 'cosine', or 'overlap')
            
        Returns:
            DataFrame with similarity scores
        """
        # First check if we have the co-occurrence relationships
        count_query = f"""
        MATCH ()-[r:`{co_occurs_rel}`]->()
        RETURN count(r) AS rel_count
        """
        count_result = self.run_cypher_query(count_query)
        rel_count = count_result.iloc[0]['rel_count'] if not count_result.empty else 0
        
        print(f"Found {rel_count} {co_occurs_rel} relationships to analyze")
        
        if rel_count == 0:
            print(f"No {co_occurs_rel} relationships found. Try creating the co-occurrence network first.")
            return pd.DataFrame()
        
        # Define limit clause
        limit_clause = f"LIMIT {top_k}" if top_k > 0 else ""
        bottom_clause = f"LIMIT {bottom_k}" if bottom_k > 0 else ""
        
        # Use direct relationship similarity if already have co-occurrence relationships
        if rel_count > 0:
            # Ensure we only get one direction by using id(n1) < id(n2)
            query = f"""
            MATCH (n1:`{node_label}`)-[r:`{co_occurs_rel}`]-(n2:`{node_label}`)
            WHERE id(n1) < id(n2)
            WITH n1, n2, r.weight AS weight
            ORDER BY weight DESC
            {limit_clause}
            RETURN n1.name AS node1, n2.name AS node2, weight AS similarity({algorithm.capitalize()})
            """
            
            try:
                direct_df = self.run_cypher_query(query)
                if not direct_df.empty:
                    print(f"Found {len(direct_df)} similar pairs using direct relationships")
                    return direct_df
            except Exception as e:
                print(f"Error with direct similarity query: {e}")
        
        # If no direct relationships with weights or they don't return results,
        # calculate similarity based on the specified algorithm
        print(f"Calculating similarity using {algorithm} algorithm...")
        
        # Define the similarity calculation based on the algorithm
        if algorithm.lower() == 'cosine':
            # Cosine similarity: dot product of vectors divided by product of their magnitudes
            similarity_calc = """
            toFloat(shared_count) / sqrt(toFloat(n1_count) * toFloat(n2_count)) AS similarity
            """
        elif algorithm.lower() == 'overlap':
            # Overlap coefficient: size of intersection divided by size of smaller set
            similarity_calc = """
            toFloat(shared_count) / toFloat(CASE WHEN n1_count < n2_count THEN n1_count ELSE n2_count END) AS similarity
            """
        else:  # Default to Jaccard
            # Jaccard similarity: size of intersection divided by size of union
            similarity_calc = """
            toFloat(shared_count) / (n1_count + n2_count - shared_count) AS similarity
            """
        
        # Ensure we only get one direction by using id(n1) < id(n2)
        query = f"""
        MATCH (n1:`{node_label}`)-[:`{co_occurs_rel}`]-(shared)-[:`{co_occurs_rel}`]-(n2:`{node_label}`)
        WHERE id(n1) < id(n2)
        WITH n1, n2, 
             COUNT(DISTINCT shared) AS shared_count,
             SIZE((n1)-[:`{co_occurs_rel}`]-()) AS n1_count,
             SIZE((n2)-[:`{co_occurs_rel}`]-()) AS n2_count
        WITH n1, n2, shared_count, n1_count, n2_count,
             {similarity_calc}
        WHERE similarity > 0  
        RETURN n1.name AS node1, n2.name AS node2, similarity AS similarity({algorithm.capitalize()})
        ORDER BY similarity({algorithm.capitalize()}) DESC
        {limit_clause}
        {bottom_clause}
        """
        
        try:
            df = self.run_cypher_query(query)
            if df.empty:
                print(f"No similarities found with {algorithm} calculation")
                
                # Try one more approach - just count shared connections
                simple_query = f"""
                MATCH (n1:`{node_label}`)-[:`{co_occurs_rel}`]-(shared)-[:`{co_occurs_rel}`]-(n2:`{node_label}`)
                WHERE id(n1) < id(n2)
                WITH n1, n2, COUNT(DISTINCT shared) AS shared_count
                WHERE shared_count > 0
                RETURN n1.name AS node1, n2.name AS node2, shared_count AS similarity({algorithm.capitalize()})
                ORDER BY similarity({algorithm.capitalize()}) DESC
                {limit_clause}
                {bottom_clause}
                """
                
                df = self.run_cypher_query(simple_query)
                if not df.empty:
                    print(f"Found {len(df)} similar pairs using shared count")
                else:
                    print("No similarities found with any method. The graph may not have enough connections.")
            else:
                print(f"Found {len(df)} similar pairs using {algorithm} similarity")
            return df
        except Exception as e:
            print(f"Error calculating node similarity: {e}")
            return pd.DataFrame()
            
    def calculate_link_prediction(self, node_label, co_occurs_rel='CO_OCCURS_WITH', method='jaccard', top_k=10):
        """
        Predict links based on similarity metrics without using GDS.
        This is an alternative when GDS link prediction is not available.
        
        Args:
            node_label: Node type to analyze
            co_occurs_rel: Co-occurrence relationship type
            method: Similarity method ('jaccard' or 'adamic_adar')
            top_k: Number of top results to return
            
        Returns:
            DataFrame with predicted links and scores
        """
        # For nodes that aren't directly connected but have shared connections,
        # calculate a similarity score to predict potential links
        
        if method == 'jaccard':
            similarity_calc = """
            toFloat(shared_count) / (n1_count + n2_count - shared_count) AS score
            """
        elif method == 'adamic_adar':
            # Adamic/Adar gives more weight to uncommon connections
            similarity_calc = """
            SUM(1.0 / LOG10(SIZE((shared)-[:`CO_OCCURS_WITH`]-()))) AS score
            """
        else:
            # Default to Common Neighbors
            similarity_calc = """
            shared_count AS score
            """
            
        # Add LIMIT clause only if top_k > 0
        limit_sql = f"LIMIT {top_k}" if top_k > 0 else ""
        
        # Ensure consistent ordering with n1.name < n2.name to prevent duplicate pairs
        query = f"""
        MATCH (n1:`{node_label}`)-[:`{co_occurs_rel}`]-(shared)-[:`{co_occurs_rel}`]-(n2:`{node_label}`)
        WHERE id(n1) < id(n2)
        AND NOT (n1)-[:`{co_occurs_rel}`]-(n2)  // Only predict for non-connected nodes
        WITH n1, n2, COLLECT(DISTINCT shared) AS shared_nodes,
             COUNT(DISTINCT shared) AS shared_count,
             SIZE((n1)-[:`{co_occurs_rel}`]-()) AS n1_count,
             SIZE((n2)-[:`{co_occurs_rel}`]-()) AS n2_count
        WITH n1, n2, shared_nodes, shared_count, n1_count, n2_count,
             {similarity_calc}
        WHERE n1.name < n2.name  // Extra check to ensure consistent ordering
        RETURN n1.name AS node1, n2.name AS node2, score
        ORDER BY score DESC
        {limit_sql}
        """
        
        try:
            df = self.run_cypher_query(query)
            return df
        except Exception as e:
            print(f"Error calculating link predictions: {e}")
            return pd.DataFrame()

    def list_cooccurrence_networks(self, delete=False):
        """
        List all co-occurrence networks (relationships with 'CO_OCCURS_WITH' in their type)
        and optionally delete them.
        
        Args:
            delete: If True, delete all co-occurrence relationships
            
        Returns:
            DataFrame with co-occurrence relationship statistics
        """
        # Query to find all relationships that might be co-occurrence networks
        # (looking for relationships with names containing CO_OCCURS or similar)
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        WHERE relationshipType CONTAINS 'CO_OCCURS' OR 
              relationshipType CONTAINS 'OCCURS_WITH' OR
              relationshipType CONTAINS 'COOCCURS'
        WITH relationshipType
        MATCH ()-[r:`${relationshipType}`]->()
        RETURN relationshipType AS relType, count(r) AS count
        ORDER BY count DESC
        """
        
        try:
            df = self.run_cypher_query(query)
            
            if df.empty:
                print("No co-occurrence networks found.")
                return pd.DataFrame()
            
            print("Co-occurrence networks found:")
            for _, row in df.iterrows():
                rel_type = row['relType']
                count = row['count']
                print(f"  {rel_type}: {count} relationships")
                
                # Get sample relationships for each type
                sample_query = f"""
                MATCH (a)-[r:`{rel_type}`]-(b)
                RETURN a.name AS node1, b.name AS node2, r.weight AS weight
                ORDER BY r.weight DESC
                LIMIT 3
                """
                
                try:
                    samples = self.run_cypher_query(sample_query)
                    if not samples.empty:
                        print("    Sample relationships:")
                        for _, sample in samples.iterrows():
                            weight = sample['weight'] if 'weight' in sample else 'N/A'
                            print(f"      {sample['node1']} -- {weight} --> {sample['node2']}")
                except Exception as e:
                    print(f"    Error getting samples: {e}")
            
            # Delete if requested
            if delete:
                deleted_count = 0
                for rel_type in df['relType']:
                    delete_query = f"""
                    MATCH ()-[r:`{rel_type}`]->()
                    DELETE r
                    RETURN count(*) AS deleted
                    """
                    
                    try:
                        result = self.connector.execute_query(delete_query)
                        deleted = result[0]["deleted"] if result else 0
                        deleted_count += deleted
                        print(f"Deleted {deleted} relationships of type {rel_type}")
                    except Exception as e:
                        print(f"Error deleting {rel_type} relationships: {e}")
                
                print(f"Total deleted: {deleted_count} relationships")
            
            return df
        except Exception as e:
            print(f"Error listing co-occurrence networks: {e}")
            return pd.DataFrame()

    def list_gds_projections(self, delete=False):
        """
        List all GDS graph projections and optionally delete them.
        
        Args:
            delete: If True, delete all GDS projections
            
        Returns:
            DataFrame with GDS projection statistics
        """
        # Check if GDS is available
        if not self.is_gds_available():
            print("GDS is not available. Cannot list projections.")
            return pd.DataFrame({'message': ['GDS is not available']})
        
        # Query to list all GDS projections
        try:
            # Format date as ISO format without milliseconds
            query = """
            CALL gds.graph.list()
            YIELD graphName, nodeCount, relationshipCount, creationTime
            RETURN graphName, nodeCount, relationshipCount, 
                   datetime(creationTime).year + '-' + 
                   CASE WHEN datetime(creationTime).month < 10 THEN '0' + datetime(creationTime).month ELSE datetime(creationTime).month END + '-' + 
                   CASE WHEN datetime(creationTime).day < 10 THEN '0' + datetime(creationTime).day ELSE datetime(creationTime).day END + 'T' + 
                   CASE WHEN datetime(creationTime).hour < 10 THEN '0' + datetime(creationTime).hour ELSE datetime(creationTime).hour END + ':' + 
                   CASE WHEN datetime(creationTime).minute < 10 THEN '0' + datetime(creationTime).minute ELSE datetime(creationTime).minute END + ':' + 
                   CASE WHEN datetime(creationTime).second < 10 THEN '0' + datetime(creationTime).second ELSE datetime(creationTime).second END AS created
            """
            
            df = self.run_cypher_query(query)
            
            if df.empty:
                print("No GDS projections found.")
                return pd.DataFrame({'message': ['No GDS projections found']})
            
            print(f"Found {len(df)} GDS projections:")
            for _, row in df.iterrows():
                graph_name = row['graphName']
                node_count = row['nodeCount']
                rel_count = row['relationshipCount']
                created = row['created']
                print(f"  {graph_name}: {node_count} nodes, {rel_count} relationships, created {created}")
                
                # Get more details - use basic query to avoid errors with different GDS versions
                try:
                    detail_query = f"""
                    CALL gds.graph.list('{graph_name}')
                    YIELD graphName
                    RETURN graphName
                    """
                    
                    self.run_cypher_query(detail_query)
                    print(f"    Details available via gds.graph.list('{graph_name}')")
                except Exception as e:
                    print(f"    Error getting details: {e}")
            
            # Delete if requested
            if delete:
                deleted_count = 0
                for graph_name in df['graphName']:
                    try:
                        delete_query = f"""
                        CALL gds.graph.drop('{graph_name}')
                        YIELD graphName
                        RETURN graphName
                        """
                        
                        result = self.connector.execute_query(delete_query)
                        if result:
                            deleted_count += 1
                            print(f"Deleted GDS projection: {graph_name}")
                    except Exception as e:
                        print(f"Error deleting projection {graph_name}: {e}")
                
                print(f"Total deleted: {deleted_count} GDS projections")
            
            return df
        except Exception as e:
            print(f"Error listing GDS projections: {e}")
            return pd.DataFrame({'error': [str(e)]}) 

    def run_node2vec_link_prediction(self, nx_graph, test_frac=0.3, random_state=42):
        """
        Perform link prediction using Node2Vec embeddings and a logistic regression classifier.
        Legacy function for backward compatibility.
        
        Args:
            nx_graph: networkx.Graph (undirected, e.g., actor-actor or movie-movie)
            test_frac: fraction of edges to use for testing
            random_state: random seed for reproducibility
            
        Returns: 
            y_true, y_scores (for evaluation), and dataframe with predicted links
        """
        return self.run_ml_link_prediction(nx_graph, algorithm='logistic_regression', 
                                         test_frac=test_frac, random_state=random_state)