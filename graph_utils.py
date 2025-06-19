import os
import dotenv
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import pandas as pd

class Neo4jConnector:
    def __init__(self, uri=None, user=None, password=None):
        # Load environment variables if not provided directly
        dotenv.load_dotenv()
        
        # Use provided credentials or get from environment
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        
        self.driver = None
    
    def connect(self):
        """Connect to the Neo4j database"""
        if not self.driver:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.user, self.password)
                )
                print(f"Connected to Neo4j database at {self.uri}")
                return True
            except Exception as e:
                print(f"Failed to connect to Neo4j: {e}")
                return False
        return True
    
    def close(self):
        """Close the connection to Neo4j"""
        if self.driver:
            self.driver.close()
            print("Connection to Neo4j closed")
            self.driver = None
    
    def execute_query(self, query, params=None):
        """Execute a Cypher query and return results"""
        if not self.connect():
            return None
        
        try:
            with self.driver.session() as session:
                result = session.run(query, params or {})
                return list(result)
        except Exception as e:
            print(f"Query execution failed: {e}")
            return None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class GraphVisualizer:
    """
    Utility class for creating visualizations of graph data from Neo4j.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.graph = nx.MultiDiGraph()
        self.node_colors = {
            'Movie': '#ff9999',  # Light red
            'Person': '#99ccff',  # Light blue
            'Genre': '#99ff99',   # Light green
            'Company': '#ffcc99', # Light orange
            'Country': '#cc99ff'  # Light purple
        }
    
    def create_graph_from_neo4j_results(self, results):
        """
        Create a NetworkX graph from Neo4j query results.
        
        Args:
            results: List of Neo4j Records
        """
        # Clear any existing graph
        self.graph.clear()
        
        # Process each record
        for record in results:
            # Process all items in the record
            for value in record.values():
                # Check if this is a node
                if hasattr(value, 'labels') and hasattr(value, 'element_id'):
                    self._add_node_to_graph(value)
                
                # Check if this is a relationship
                elif hasattr(value, 'type') and hasattr(value, 'start_node') and hasattr(value, 'end_node'):
                    self._add_relationship_to_graph(value)
    
    def _add_node_to_graph(self, node):
        """Add a Neo4j node to the NetworkX graph."""
        node_id = node.element_id
        
        # Skip if node already exists
        if node_id in self.graph.nodes:
            return
        
        # Extract node properties
        props = dict(node.items())
        
        # Get the primary label (first one)
        label = next(iter(node.labels), "Unknown")
        
        # Get display name
        if 'name' in props:
            display_name = props['name']
        elif 'title' in props:
            display_name = props['title']
        else:
            display_name = str(node_id)
        
        # Add node to graph
        self.graph.add_node(
            node_id,
            label=label,
            name=display_name,
            properties=props
        )
    
    def _add_relationship_to_graph(self, relationship):
        """Add a Neo4j relationship to the NetworkX graph."""
        start_id = relationship.start_node.element_id
        end_id = relationship.end_node.element_id
        rel_type = relationship.type
        
        # Add nodes if they don't exist
        if start_id not in self.graph.nodes:
            self._add_node_to_graph(relationship.start_node)
        
        if end_id not in self.graph.nodes:
            self._add_node_to_graph(relationship.end_node)
        
        # Add the edge
        props = dict(relationship.items())
        self.graph.add_edge(
            start_id, 
            end_id, 
            type=rel_type,
            properties=props
        )
    
    def visualize_with_matplotlib(self, title=None, layout='spring'):
        """
        Visualize the graph using matplotlib.
        
        Args:
            title: Title for the plot
            layout: Layout algorithm ('spring', 'circular', 'shell', 'kamada_kawai')
        """
        if len(self.graph) == 0:
            print("No graph to visualize")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Choose layout algorithm
        if layout == 'spring':
            pos = nx.spring_layout(self.graph)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'shell':
            pos = nx.shell_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Get node labels, colors, and sizes
        node_labels = {node: data['name'] for node, data in self.graph.nodes(data=True)}
        node_colors = [self.node_colors.get(data['label'], '#cccccc') for _, data in self.graph.nodes(data=True)]
        
        # Set node sizes based on degree
        node_sizes = [300 + 100 * self.graph.degree(node) for node in self.graph.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                               node_color=node_colors, 
                               node_size=node_sizes,
                               alpha=0.8)
        
        # Draw edges with different colors per type
        edge_types = list(set([data['type'] for _, _, data in self.graph.edges(data=True)]))
        edge_colors = plt.cm.tab10(range(len(edge_types)))
        edge_type_to_color = {edge_type: edge_colors[i] for i, edge_type in enumerate(edge_types)}
        
        for edge_type in edge_types:
            edges = [(u, v) for u, v, data in self.graph.edges(data=True) if data['type'] == edge_type]
            if edges:
                nx.draw_networkx_edges(self.graph, pos, 
                                       edgelist=edges, 
                                       width=1.5, 
                                       alpha=0.6,
                                       edge_color=[edge_type_to_color[edge_type]] * len(edges),
                                       label=edge_type)
        
        # Draw node labels if graph is not too large
        if len(self.graph) <= 50:
            nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=8)
        
        # Create legend for node types
        node_types = list(set([data['label'] for _, data in self.graph.nodes(data=True)]))
        node_legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=self.node_colors.get(nt, '#cccccc'), 
                                          markersize=10, label=nt) 
                               for nt in node_types]
        
        plt.legend(handles=node_legend_elements, title="Node Types", loc='upper left')
        
        # Set title
        if title:
            plt.title(title)
        else:
            plt.title(f"Graph Visualization ({len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges)")
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save the figure
        filename = title.replace(' ', '_').lower() + '.png' if title else 'graph_visualization.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Static visualization saved as {filename}")
        
        plt.show()
    
    def save_interactive_html(self, filename='interactive_graph.html', height='750px', bgcolor='#222222'):
        """
        Create and save an interactive HTML visualization using PyVis.
        
        Args:
            filename: Name of the HTML file to save
            height: Height of the visualization
            bgcolor: Background color
        """
        if len(self.graph) == 0:
            print("No graph to visualize")
            return
        
        # Create PyVis network
        net = Network(height=height, width='100%', bgcolor=bgcolor, font_color='white')
        
        # Add nodes
        for node_id, data in self.graph.nodes(data=True):
            label = data['label']
            title = f"{label}: {data['name']}"
            
            # Add properties to title tooltip
            if 'properties' in data:
                for k, v in data['properties'].items():
                    if k not in ['name', 'title']:
                        title += f"<br>{k}: {v}"
            
            net.add_node(node_id, 
                        label=data['name'], 
                        title=title, 
                        color=self.node_colors.get(label, '#cccccc'),
                        group=label)
        
        # Add edges
        for u, v, data in self.graph.edges(data=True):
            relation = data['type']
            
            # Create edge title with properties
            title = relation
            if 'properties' in data:
                for k, v in data['properties'].items():
                    title += f"<br>{k}: {v}"
            
            net.add_edge(u, v, title=title, label=relation)
        
        # Set options
        net.set_options("""
        var options = {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "solver": "forceAtlas2Based",
                "stabilization": {
                    "iterations": 100
                }
            },
            "edges": {
                "arrows": {
                    "to": {
                        "enabled": true
                    }
                },
                "color": {
                    "inherit": true
                },
                "smooth": {
                    "enabled": false
                }
            },
            "interaction": {
                "hover": true,
                "navigationButtons": true,
                "tooltipDelay": 100
            }
        }
        """)
        
        # Save the visualization
        net.save_graph(filename)
        print(f"Interactive visualization saved as {filename}")
    
    def export_to_graphml(self, filename='graph_export.graphml'):
        """
        Export the graph to GraphML format for use in other tools.
        
        Args:
            filename: Name of the GraphML file to save
        """
        if len(self.graph) == 0:
            print("No graph to export")
            return
        
        nx.write_graphml(self.graph, filename)
        print(f"Graph exported to {filename}")
    
    def get_network_statistics(self):
        """
        Calculate basic graph statistics.
        
        Returns:
            dict: Dictionary of graph statistics
        """
        if len(self.graph) == 0:
            return {"error": "No graph to analyze"}
        
        stats = {
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "node_types": dict(pd.Series([data['label'] for _, data in self.graph.nodes(data=True)]).value_counts()),
            "edge_types": dict(pd.Series([data['type'] for _, _, data in self.graph.edges(data=True)]).value_counts()),
            "density": nx.density(self.graph)
        }
        
        # Add more advanced metrics for smaller graphs
        if len(self.graph) < 1000:
            undirected = self.graph.to_undirected()
            if nx.is_connected(undirected):
                stats["avg_shortest_path"] = nx.average_shortest_path_length(undirected)
                stats["diameter"] = nx.diameter(undirected)
            
            stats["avg_clustering"] = nx.average_clustering(undirected)
        
        return stats 