#!/usr/bin/env python3
"""
Knowledge Graph Completion with PyKEEN from a Neo4j backend.
This file provides the PykeenMovieKG class for use with the Streamlit app (app.py).
"""

import pickle
import os
import numpy as np # Added NumPy import
from neo4j import GraphDatabase # Retained for type hinting if connector is not fully typed
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
# Assuming Neo4jConnector is a class that provides a self.driver attribute
# and potentially self.uri, self.user, self.password, self.database
# from graph_utils import Neo4jConnector # For type hinting if available

class PykeenMovieKG:
    def __init__(self, connector): # connector is an instance of Neo4jConnector
        """
        Initializes the PykeenMovieKG class with a Neo4j connector.

        Args:
            connector: An instance of Neo4jConnector (or a similar class
                       that provides a .driver attribute for Neo4j connections).
        """
        self.connector = connector
        self.triples: list[tuple[str, str, str]] = []
        self.factory: TriplesFactory | None = None
        self.models: dict[str, object] = {} # Stores trained models in memory

        # Attempt to get database from connector if available, otherwise default to None
        self.neo4j_database: str | None = getattr(connector, 'database', None)
        if hasattr(connector, 'uri'): # For logging or direct use if needed
            print(f"PykeenMovieKG initialized with connector for URI: {connector.uri}")


    def _get_default_cypher_query(self) -> str:
        """
        Returns the default Cypher query for fetching triples.
        This query is designed to be robust for various node types in a movie graph.
        """
        return """
            MATCH (h)-[r]->(t)
            WHERE h IS NOT NULL AND t IS NOT NULL AND type(r) IS NOT NULL
            RETURN
                CASE
                    WHEN 'Movie' IN labels(h) THEN h.title
                    WHEN 'Person' IN labels(h) THEN h.name
                    WHEN 'Genre' IN labels(h) THEN h.name
                    WHEN 'Company' IN labels(h) THEN h.name
                    WHEN 'Country' IN labels(h) THEN h.name
                    ELSE COALESCE(h.name, h.title, toString(id(h))) // Fallback
                END AS head_name,
                type(r) AS relation,
                CASE
                    WHEN 'Movie' IN labels(t) THEN t.title
                    WHEN 'Person' IN labels(t) THEN t.name
                    WHEN 'Genre' IN labels(t) THEN t.name
                    WHEN 'Company' IN labels(t) THEN t.name
                    WHEN 'Country' IN labels(t) THEN t.name
                    ELSE COALESCE(t.name, t.title, toString(id(t))) // Fallback
                END AS tail_name
            LIMIT 100000 // Limit to manage performance for very large graphs
        """

    def load_triples_from_neo4j(self, cypher_query: str | None = None) -> bool:
        """
        Connects to Neo4j using the provided connector, executes a Cypher query
        returning head_name, relation, tail_name, and collects results.
        Populates self.triples and self.factory.

        Args:
            cypher_query (str, optional): The Cypher query to execute.
                                           Defaults to _get_default_cypher_query().

        Returns:
            bool: True if triples were successfully loaded, False otherwise.
        """
        query_to_run = cypher_query if cypher_query else self._get_default_cypher_query()
        loaded_triples: list[tuple[str, str, str]] = []

        print(f"PykeenMovieKG: Loading triples using database: {self.neo4j_database or 'default'}")
        try:
            # Assumes self.connector.driver is a valid Neo4j driver instance
            with self.connector.driver.session(database=self.neo4j_database) as session:
                result = session.run(query_to_run)
                for record in result:
                    h = record.get("head_name")
                    r = record.get("relation")
                    t = record.get("tail_name")
                    if h is not None and r is not None and t is not None:
                        loaded_triples.append((str(h), str(r), str(t)))
                    else:
                        print(f"PykeenMovieKG: Warning - Skipped triple with missing components: h={h}, r={r}, t={t}")

            self.triples = loaded_triples
            if self.triples:
                print(f"PykeenMovieKG: Successfully loaded {len(self.triples)} triples.")
                print(f"PykeenMovieKG: Example triple: {self.triples[0]}")
                # Convert list of tuples to NumPy array before creating TriplesFactory
                np_triples = np.array(self.triples, dtype=str)
                self.factory = TriplesFactory.from_labeled_triples(np_triples)
                return True
            else:
                print("PykeenMovieKG: No triples were loaded from Neo4j.")
                self.factory = None
                return False
        except Exception as e:
            print(f"PykeenMovieKG: Error loading triples from Neo4j: {e}")
            import traceback
            traceback.print_exc()
            self.triples = []
            self.factory = None
            return False

    def prepare_training_data(self) -> bool:
        """
        Prepares training data by loading triples from Neo4j.
        This is the primary method app.py calls to initiate data loading.

        Returns:
            bool: True if data preparation was successful, False otherwise.
        """
        print("PykeenMovieKG: Preparing training data...")
        return self.load_triples_from_neo4j() # Uses default query

    def train_model(
        self,
        model_name: str,
        epochs: int = 100,
        embedding_dim: int = 100,
        device: str = "cpu",
        save_model_flag: bool = True, # Renamed from save_model to avoid conflict with method
        model_output_dir: str = "models"
    ) -> object | None:
        """
        Trains a specific PyKEEN model.

        Args:
            model_name (str): Name of the PyKEEN model (e.g., "TransE", "RotatE").
            epochs (int): Number of training epochs.
            embedding_dim (int): Dimensionality of the embeddings.
            device (str): Device to train on ("cpu" or "cuda").
            save_model_flag (bool): Whether to save the trained model to disk.
            model_output_dir (str): Directory to save the model.

        Returns:
            object | None: The trained PyKEEN model object, or None if training failed.
        """
        if not self.factory:
            print(f"PykeenMovieKG: TriplesFactory not available for training {model_name}.")
            print("PykeenMovieKG: Attempting to prepare training data now...")
            if not self.prepare_training_data() or not self.factory:
                print(f"PykeenMovieKG: Failed to prepare training data. Cannot train {model_name}.")
                return None
            print("PykeenMovieKG: Training data prepared successfully. Proceeding with training.")

        print(f"PykeenMovieKG: Training {model_name} for {epochs} epochs (dim: {embedding_dim}) on {device}...")

        model_kwargs = {'embedding_dim': embedding_dim}

        try:
            # Split the factory into training and testing sets
            # Using a 80/20 split for training/testing. Adjust ratios as needed.
            # PyKEEN's split method returns a list of TriplesFactories.
            training_factory, testing_factory = self.factory.split([0.8, 0.2], random_state=42)

            result = pipeline(
                training=training_factory, # Pass the training part
                testing=testing_factory,  # Pass the testing part
                model=model_name,
                model_kwargs=model_kwargs,
                epochs=epochs,
                device=device,
                random_seed=42,  # For reproducibility
                # Optional: Add early stopping if desired
                # stopper='early',
                # stopper_kwargs=dict(frequency=10, patience=3, delta=0.001, metric='hits@10')
            )
        except Exception as e:
            print(f"PykeenMovieKG: Error during PyKEEN pipeline for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

        trained_kge_model = result.model
        self.models[model_name] = trained_kge_model  # Store in memory

        if result.metric_results:
            try:
                mrr = result.get_metric('mean_reciprocal_rank')
                hits_at_10 = result.get_metric('hits@10')
                print(f"PykeenMovieKG: Trained {model_name} -> MRR: {mrr:.4f}, Hits@10: {hits_at_10:.4f}")
            except KeyError: # If some metrics are not available
                print(f"PykeenMovieKG: Trained {model_name}. Some metrics might not be available for this model/dataset size.")
        else:
            print(f"PykeenMovieKG: Trained {model_name}. Metrics not available (e.g., if training was skipped or failed).")

        if save_model_flag:
            # Filename convention matches app.py's check: models/pykeen_<model_name_lower>_model.pkl
            model_filename = f"pykeen_{model_name.lower()}_model.pkl"
            model_filepath = os.path.join(model_output_dir, model_filename)
            self._save_trained_model_to_disk(trained_kge_model, model_filepath)
        
        return trained_kge_model

    def _save_trained_model_to_disk(self, model: object, filepath: str) -> None:
        """
        Serializes a trained PyKEEN model to the given filepath using pickle.
        Helper method for train_model.
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as out_file:
                pickle.dump(model, out_file)
            print(f"PykeenMovieKG: Model successfully saved to {filepath}")
        except Exception as e:
            print(f"PykeenMovieKG: Error saving model to {filepath}: {e}")
            import traceback
            traceback.print_exc()

# Example of how this class might be used (for testing, not part of app.py's direct flow):
if __name__ == "__main__":
    print("Running PykeenMovieKG standalone example (requires Neo4j connection details)")

    # This example requires you to have a Neo4jConnector-like class
    # or adapt it to use direct GraphDatabase.driver
    
    # Dummy Neo4jConnector for testing purposes
    class DummyConnector:
        def __init__(self, uri, user, password, database=None):
            self.uri = uri
            self.user = user
            self.password = password
            self.database = database
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                self.driver.verify_connectivity()
                print("DummyConnector: Successfully connected to Neo4j.")
            except Exception as e:
                print(f"DummyConnector: Failed to connect to Neo4j at {uri} - {e}")
                self.driver = None

        def close(self):
            if self.driver:
                self.driver.close()
                print("DummyConnector: Connection closed.")

    # --- Configuration for standalone example ---
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password" # Replace with your password
    NEO4J_DB = None # Or your specific database name

    print(f"Attempting to connect to Neo4j at {NEO4J_URI}...")
    test_connector = DummyConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB)

    if test_connector.driver:
        kg_completer = PykeenMovieKG(test_connector)

        # 1. Prepare training data (loads triples)
        print("\n--- Preparing Training Data ---")
        if kg_completer.prepare_training_data():
            print(f"Number of triples loaded: {len(kg_completer.triples)}")

            # 2. Train a model (e.g., TransE)
            print("\n--- Training TransE Model ---")
            transe_model = kg_completer.train_model(
                model_name="TransE",
                epochs=5, # Reduced for quick testing
                embedding_dim=50, # Reduced for quick testing
                save_model_flag=True,
                model_output_dir="models_test" # Use a test directory
            )
            if transe_model:
                print("TransE model training example finished.")
            else:
                print("TransE model training example failed.")
            
            # Example: Train another model (e.g., ComplEx)# print("\n--- Training ComplEx Model ---")# complex_model = kg_completer.train_model("ComplEx", epochs=5, embedding_dim=50, save_model_flag=False)# if complex_model:#    print("ComplEx model trained and stored in memory.")

        else:
            print("Failed to prepare training data. Aborting example.")
        
        test_connector.close()
    else:
        print("Cannot run example without a Neo4j connection.")

    print("\nStandalone example finished.")
