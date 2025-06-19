import os
import pickle
import numpy as np
import pandas as pd
import torch
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Version identifier
__version__ = "2.0.0"

# PyKEEN imports
try:
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline
    from pykeen.models import TransE, PairRE, RotatE, ComplEx
    from pykeen.evaluation import RankBasedEvaluator
    from pykeen.training import SLCWATrainingLoop, LCWATrainingLoop
    PYKEEN_AVAILABLE = True
except ImportError:
    PYKEEN_AVAILABLE = False
    logger.error("PyKEEN not available. Install it with: pip install pykeen>=1.10.3")

class PykeenMovieKG:
    """
    Knowledge Graph Completion using PyKEEN
    
    This class provides functionality to:
    1. Load and process triple data
    2. Train knowledge graph embedding models
    3. Make predictions for missing nodes or relationships
    4. Evaluate model performance
    """
    
    SUPPORTED_MODELS = {
        "TransE": TransE,
        "PairRE": PairRE,
        "RotatE": RotatE,
        "ComplEx": ComplEx
    }
    
    def __init__(self, neo4j_connector=None):
        """
        Initialize the KG completion module
        
        Args:
            neo4j_connector: Optional Neo4j connector for retrieving data from a database
                            (default: None, data will be loaded from files)
        """
        self.connector = neo4j_connector
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.relation_to_id = {}
        self.id_to_relation = {}
        self.triples_factory = None
        self.train_triples_factory = None
        self.test_triples_factory = None
        self.model = None
        self.model_name = None
        self.triples = []
        self.pipeline_result = None
        self.factory = None  # Alias for triples_factory for compatibility
        
    def load_triples_from_file(self, filepath: str, delimiter: str = '\t') -> bool:
        """
        Load triples from a tab-separated file in (head, relation, tail) format
        
        Args:
            filepath: Path to the file containing triples
            delimiter: Column delimiter in the file (default: tab)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading triples from {filepath}")
            
            # Check if file exists
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return False
            
            # Read triples from file
            triples_data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split(delimiter)
                        if len(parts) >= 3:
                            h, r, t = parts[0], parts[1], parts[2]
                            triples_data.append((h, r, t))
            
            if not triples_data:
                logger.error("No valid triples found in the file")
                return False
            
            # Convert to numpy array
            self.triples = triples_data
            triples_array = np.array(triples_data)
            
            # Create PyKEEN TriplesFactory
            self.triples_factory = TriplesFactory.from_labeled_triples(triples_array)
            self.factory = self.triples_factory  # Set alias for compatibility
            
            # Store mappings
            self.entity_to_id = self.triples_factory.entity_to_id
            # Handle the case when entity_to_id is a dict without an inverse attribute
            if hasattr(self.entity_to_id, 'inverse'):
                self.id_to_entity = self.entity_to_id.inverse
            else:
                # Manually create inverse mapping
                self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
                
            self.relation_to_id = self.triples_factory.relation_to_id
            # Handle the case when relation_to_id is a dict without an inverse attribute
            if hasattr(self.relation_to_id, 'inverse'):
                self.id_to_relation = self.relation_to_id.inverse
            else:
                # Manually create inverse mapping
                self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}
            
            logger.info(f"Loaded {len(self.triples)} triples with {len(self.entity_to_id)} entities and {len(self.relation_to_id)} relation types")
            return True
            
        except Exception as e:
            logger.error(f"Error loading triples from file: {e}")
            traceback.print_exc()
            return False
    
    def prepare_training_data(self, **kwargs) -> bool:
        """
        Extract triples from a database using the provided connector
        
        Args:
            **kwargs: Arguments to pass to the connector (e.g. max_triples)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.connector is None:
                logger.error("No database connector provided")
                return False
                
            # Get max_triples from kwargs if available
            max_triples = kwargs.get('max_triples', 10000)
            
            logger.info(f"Extracting up to {max_triples} triples from database")
            
            # First, let's get an estimate of how many triples are in the database
            count_query = """
            MATCH ()-[r]->() 
            RETURN count(r) as total_rel
            """
            count_result = self.connector.execute_query(count_query)
            total_relationships = count_result[0]["total_rel"] if count_result else 0
            
            logger.info(f"Found approximately {total_relationships} relationships in Neo4j")
            
            # Extract triples from Neo4j with a comprehensive query
            query = f"""
            // Person-Movie relationships
            MATCH (h:Person)-[r]->(t:Movie)
            RETURN h.name AS head, type(r) AS relation, t.title AS tail
            UNION
            // Movie-Person relationships
            MATCH (h:Movie)-[r]->(t:Person)
            RETURN h.title AS head, type(r) AS relation, t.name AS tail
            UNION
            // Movie-Movie relationships
            MATCH (h:Movie)-[r]->(t:Movie)
            RETURN h.title AS head, type(r) AS relation, t.title AS tail
            UNION
            // Movie-Genre relationships
            MATCH (h:Movie)-[r]->(t:Genre)
            RETURN h.title AS head, type(r) AS relation, t.name AS tail
            UNION
            // Genre-Movie relationships
            MATCH (h:Genre)-[r]->(t:Movie)
            RETURN h.name AS head, type(r) AS relation, t.title AS tail
            UNION
            // Person-Person relationships
            MATCH (h:Person)-[r]->(t:Person)
            RETURN h.name AS head, type(r) AS relation, t.name AS tail
            LIMIT {max_triples}
            """
            
            triples_data = self.connector.execute_query(query)
            
            if not triples_data:
                logger.error("No triples found in database")
                return False
            
            # Convert to a list of tuples
            self.triples = [(row.get("head"), row.get("relation"), row.get("tail")) 
                           for row in triples_data 
                           if row.get("head") is not None 
                           and row.get("relation") is not None 
                           and row.get("tail") is not None]
            
            # Remove duplicates
            self.triples = list(set(self.triples))
            
            # Convert to numpy array
            triples_array = np.array(self.triples)
            
            # Create PyKEEN TriplesFactory
            self.triples_factory = TriplesFactory.from_labeled_triples(triples_array)
            self.factory = self.triples_factory  # Set alias for compatibility
            
            # Store mappings
            self.entity_to_id = self.triples_factory.entity_to_id
            # Handle the case when entity_to_id is a dict without an inverse attribute
            if hasattr(self.entity_to_id, 'inverse'):
                self.id_to_entity = self.entity_to_id.inverse
            else:
                # Manually create inverse mapping
                self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
                
            self.relation_to_id = self.triples_factory.relation_to_id
            # Handle the case when relation_to_id is a dict without an inverse attribute
            if hasattr(self.relation_to_id, 'inverse'):
                self.id_to_relation = self.relation_to_id.inverse
            else:
                # Manually create inverse mapping
                self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}
            
            logger.info(f"Extracted {len(self.triples)} unique triples with {len(self.entity_to_id)} entities and {len(self.relation_to_id)} relation types")
            
            # Split data into train/test if not already done
            if not hasattr(self, 'train_triples_factory') or self.train_triples_factory is None:
                logger.info("Automatically splitting data into training (80%) and testing (20%) sets")
                self.train_triples_factory, self.test_triples_factory = self.triples_factory.split([0.8, 0.2], random_state=42)
            
            return True
            
        except Exception as e:
            logger.error(f"Error extracting triples from database: {e}")
            traceback.print_exc()
            return False
    
    def split_train_test(self, test_ratio: float = 0.2, random_state: int = 42) -> bool:
        """
        Split the triples into training and testing sets
        
        Args:
            test_ratio: Proportion of triples to use for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.triples_factory is None:
                logger.error("No triples loaded. Load triples first.")
                return False
                
            logger.info(f"Splitting triples into train/test with ratio {1-test_ratio}/{test_ratio}")
            
            # Split the factory
            self.train_triples_factory, self.test_triples_factory = self.triples_factory.split(
                [1 - test_ratio, test_ratio],
                random_state=random_state
            )
            
            logger.info(f"Training set: {self.train_triples_factory.num_triples} triples")
            logger.info(f"Testing set: {self.test_triples_factory.num_triples} triples")
            
            return True
            
        except Exception as e:
            logger.error(f"Error splitting triples: {e}")
            traceback.print_exc()
            return False
    
    def train_model(
        self,
        model_name: str = "TransE",
        epochs: int = 100,
        embedding_dim: int = 100,
        learning_rate: float = 0.001,
        training_loop: str = "slcwa",
        device: str = "cpu",
        save_model_flag: bool = True,
        model_output_dir: str = "models",
        batch_size: int = 128,
    ):
        """
        Train a KGE model using PyKEEN pipeline.
        """
        # Check if we have the necessary data
        if self.factory is None:
            if not self.prepare_training_data():
                logger.error("Failed to prepare training data")
                return None
                
        # Ensure we have split the data into train/test sets
        if not hasattr(self, 'train_triples_factory') or self.train_triples_factory is None:
            logger.info("Splitting data into train/test sets")
            self.train_triples_factory, self.test_triples_factory = self.factory.split([0.8, 0.2], random_state=42)

        # Device handling - try to detect CUDA if device="cuda" and provide better fallback
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
            
        # Adjust batch size if CPU (smaller for less memory usage)
        actual_batch_size = batch_size
        if device == "cpu" and batch_size > 64:
            logger.info(f"Reducing batch size from {batch_size} to 64 for CPU training")
            actual_batch_size = 64

        # 1. Map training_loop string to the TrainingLoop class (not instance)
        tl = training_loop.lower()
        if tl == "slcwa":
            training_loop_cls = SLCWATrainingLoop
        elif tl == "lcwa":
            training_loop_cls = LCWATrainingLoop
        else:
            logger.error(f"Unknown training loop: {training_loop}")
            return None

        # 2. Map model_name to the PyKEEN model class
        model_map = {
            "transe": TransE,
            "rotate": RotatE,
            "rotat e": RotatE,
            "complex": ComplEx,
            "pairre": PairRE,
        }
        key = model_name.lower()
        if key not in model_map:
            logger.error(f"Unsupported model: {model_name}")
            return None
        model_cls = model_map[key]

        # 3. Prepare model_kwargs (double dims for RotatE/ComplEx)
        dims = embedding_dim * 2 if key in ["rotate", "rotat e", "complex"] else embedding_dim
        model_kwargs = {"embedding_dim": dims}

        # 4. Ensure output directory exists
        os.makedirs(model_output_dir, exist_ok=True)

        # Set appropriate dataloader kwargs based on device

        # 5. Run PyKEEN pipeline, passing the class and optional kwargs
        logger.info(f"Starting training with {model_name} on {device} for {epochs} epochs")
        result = pipeline(
            training=self.train_triples_factory,
            testing=self.test_triples_factory,
            model=model_cls,
            model_kwargs=model_kwargs,
            optimizer_kwargs={"lr": learning_rate},
            training_loop=training_loop_cls,
            training_loop_kwargs={},
            training_kwargs={
                "num_epochs": epochs,
                "batch_size": actual_batch_size,
            },
            evaluation_kwargs={
                "batch_size": actual_batch_size,
            },
            device=device,
            random_seed=42,
        )

        # 6. Save model if requested
        if save_model_flag:
            # Save the pipeline result directory
            result.save_to_directory(model_output_dir)
            logger.info(f"Pipeline results for {model_name} saved to {model_output_dir}")
            
            # Also save just the model file with a consistent naming pattern for loading
            model_file = os.path.join(model_output_dir, f"pykeen_{model_name.lower()}_model.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(result.model, f)
            logger.info(f"Model {model_name} saved to {model_file}")

        self.pipeline_result = result
        self.model = result.model
        self.model_name = model_name
        
        # Update entity and relation mappings from the model's factory
        self.entity_to_id = result.training.entity_to_id
        if hasattr(self.entity_to_id, 'inverse'):
            self.id_to_entity = self.entity_to_id.inverse
        else:
            # Manually create inverse mapping
            self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
            
        self.relation_to_id = result.training.relation_to_id
        if hasattr(self.relation_to_id, 'inverse'):
            self.id_to_relation = self.relation_to_id.inverse
        else:
            # Manually create inverse mapping
            self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}
        
        logger.info(f"Model {model_name} trained successfully")
        
        return result
    
    def load_model(self, model_name: str, model_dir: str = "models") -> bool:
        """
        Load a trained model from disk
        
        Args:
            model_name: Name of the model to load (TransE, PairRE, etc.)
            model_dir: Directory where the model is saved
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            model_path = os.path.join(model_dir, f"pykeen_{model_name.lower()}_model.pkl")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
                
            logger.info(f"Loading model from {model_path}")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Try to infer model name from class name
            self.model_name = model_name
            
            # If entity mappings aren't available, try to recover from model
            if not self.entity_to_id and hasattr(self.model, 'triples_factory'):
                self.entity_to_id = self.model.triples_factory.entity_to_id
                if hasattr(self.entity_to_id, 'inverse'):
                    self.id_to_entity = self.entity_to_id.inverse
                else:
                    # Manually create inverse mapping
                    self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
                    
                self.relation_to_id = self.model.triples_factory.relation_to_id
                if hasattr(self.relation_to_id, 'inverse'):
                    self.id_to_relation = self.relation_to_id.inverse
                else:
                    # Manually create inverse mapping
                    self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}
                
            logger.info(f"Model {self.model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            traceback.print_exc()
            return False
    
    def _calculate_ranking_metrics(self, scores: np.ndarray, target_indices: List[int]) -> Dict[str, float]:
        """
        Calculate rank-based evaluation metrics for predictions
        
        Args:
            scores: Raw prediction scores for all candidates
            target_indices: List of indices of target entities/relations to rank
            
        Returns:
            Dict with ranking metrics
        """
        try:
            # Sort scores in descending order (higher scores = better predictions)
            # For distance-based models, we'll handle this in the calling function
            sorted_indices = np.argsort(-scores)
            
            metrics = {}
            reciprocal_ranks = []
            hits_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
            
            for target_idx in target_indices:
                # Find rank of target (1-indexed)
                rank = np.where(sorted_indices == target_idx)[0][0] + 1
                
                # Calculate reciprocal rank
                rr = 1.0 / rank
                reciprocal_ranks.append(rr)
                
                # Calculate hits@k
                for k in hits_at_k.keys():
                    if rank <= k:
                        hits_at_k[k] += 1
            
            # Calculate metrics
            num_targets = len(target_indices)
            metrics['mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
            
            for k in hits_at_k.keys():
                metrics[f'hits_at_{k}'] = hits_at_k[k] / num_targets if num_targets > 0 else 0.0
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating ranking metrics: {e}")
            return {'mrr': 0.0, 'hits_at_1': 0.0, 'hits_at_3': 0.0, 'hits_at_5': 0.0, 'hits_at_10': 0.0}
    
    def _get_ranking_info(self, scores: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get ranking information for top-k predictions
        
        Args:
            scores: Raw prediction scores for all candidates
            k: Number of top predictions to return
            
        Returns:
            List of dictionaries with ranking information
        """
        try:
            # For distance-based models, sort ascending (lower is better)
            # For similarity-based models, sort descending (higher is better)
            distance_based_models = ["TransE", "RotatE", "PairRE"]
            
            if self.model_name in distance_based_models:
                sorted_indices = np.argsort(scores)  # Ascending for distance-based
            else:
                sorted_indices = np.argsort(-scores)  # Descending for similarity-based
            
            top_k_indices = sorted_indices[:k]
            total_candidates = len(scores)
            
            ranking_info = []
            for i, idx in enumerate(top_k_indices):
                rank = i + 1
                raw_score = float(scores[idx])
                reciprocal_rank = 1.0 / rank
                rank_percentile = ((total_candidates - rank) / total_candidates) * 100
                
                ranking_info.append({
                    'index': int(idx),
                    'rank': rank,
                    'raw_score': raw_score,
                    'reciprocal_rank': reciprocal_rank,
                    'rank_percentile': rank_percentile,
                    'hits_at_1': 1 if rank == 1 else 0,
                    'hits_at_3': 1 if rank <= 3 else 0,
                    'hits_at_5': 1 if rank <= 5 else 0,
                    'hits_at_10': 1 if rank <= 10 else 0,
                })
            
            return ranking_info
            
        except Exception as e:
            logger.warning(f"Error getting ranking info: {e}")
            return []

    def predict_triplet(self, 
                       head: Optional[str] = None, 
                       relation: Optional[str] = None, 
                       tail: Optional[str] = None, 
                       k: int = 5) -> pd.DataFrame:
        """
        Predict missing part of a knowledge graph triple
        
        Provide exactly two components (head, relation, tail) and predict the missing one.
        Returns top-k predictions with ranking metrics.
        
        Args:
            head: Head entity (None to predict)
            relation: Relationship type (None to predict)
            tail: Tail entity (None to predict)
            k: Number of top predictions to return
            
        Returns:
            pd.DataFrame: DataFrame with predictions and ranking metrics
        """
        try:
            if self.model is None:
                logger.error("No model loaded. Train or load a model first.")
                return pd.DataFrame({"Error": ["No model loaded"]})
                
            # Validate that exactly one component is None
            none_count = sum(x is None for x in [head, relation, tail])
            if none_count != 1:
                error_msg = f"Exactly one of head, relation, or tail must be None. Got {none_count} None values."
                logger.error(error_msg)
                return pd.DataFrame({"Error": [error_msg]})
            
            # Validate mappings exist
            if not self.entity_to_id or not self.relation_to_id:
                error_msg = "Entity or relation mappings not available. This indicates a mismatch between training data and loaded model."
                logger.error(error_msg)
                return pd.DataFrame({"Error": [error_msg]})
                
            results = []
            
            # Predict head entities
            if head is None and relation is not None and tail is not None:
                # Validate inputs exist in training data
                if tail not in self.entity_to_id:
                    error_msg = f"Entity '{tail}' has ID {self.entity_to_id.get(tail, 'unknown')} which exceeds model's entity vocabulary size {len(self.entity_to_id)}. This indicates a mismatch between training data and loaded model."
                    logger.error(error_msg)
                    return pd.DataFrame({"Error": [error_msg]})
                    
                if relation not in self.relation_to_id:
                    error_msg = f"Relation '{relation}' not in training data. Available relations: {list(self.relation_to_id.keys())[:10]}..."
                    logger.error(error_msg)
                    return pd.DataFrame({"Error": [error_msg]})
                
                # Convert to IDs and create tensors
                tail_id = self.entity_to_id[tail]
                relation_id = self.relation_to_id[relation]
                
                # Get device from model
                device = next(self.model.parameters()).device
                
                # Try both API versions
                try:
                    # Try batch version first (older API)
                    batch = torch.tensor([[relation_id, tail_id]])
                    batch = batch.to(device)
                    predictions = self.model.predict_h(batch)
                except Exception as e:
                    logger.error(f"Error predicting head entities: {e}")
                    return pd.DataFrame({"Error": [f"Failed to predict head entities: {str(e)}"]})
                
                head_scores = predictions[0].cpu().detach().numpy()
                
                # Validate indices are within bounds
                max_entity_id = len(self.id_to_entity) - 1
                
                # Get ranking information for top-k predictions
                ranking_info = self._get_ranking_info(head_scores, k)
                
                # Convert to DataFrame with bounds checking
                for rank_info in ranking_info:
                    idx = rank_info['index']
                    if idx > max_entity_id or idx not in self.id_to_entity:
                        logger.warning(f"Skipping invalid entity index {idx} (max: {max_entity_id})")
                        continue
                        
                    entity = self.id_to_entity[idx]
                    results.append({
                        "Head": entity,
                        "Relation": relation,
                        "Tail": tail,
                        "Raw_Score": rank_info['raw_score'],
                        "Rank": rank_info['rank'],
                        "Reciprocal_Rank": rank_info['reciprocal_rank'],
                        "Rank_Percentile": rank_info['rank_percentile'],
                        "Hits@1": rank_info['hits_at_1'],
                        "Hits@3": rank_info['hits_at_3'],
                        "Hits@5": rank_info['hits_at_5'],
                        "Hits@10": rank_info['hits_at_10']
                    })
                    
            # Predict relations
            elif head is not None and relation is None and tail is not None:
                # Validate inputs exist in training data
                if head not in self.entity_to_id:
                    error_msg = f"Entity '{head}' has ID {self.entity_to_id.get(head, 'unknown')} which exceeds model's entity vocabulary size {len(self.entity_to_id)}. This indicates a mismatch between training data and loaded model."
                    logger.error(error_msg)
                    return pd.DataFrame({"Error": [error_msg]})
                    
                if tail not in self.entity_to_id:
                    error_msg = f"Entity '{tail}' has ID {self.entity_to_id.get(tail, 'unknown')} which exceeds model's entity vocabulary size {len(self.entity_to_id)}. This indicates a mismatch between training data and loaded model."
                    logger.error(error_msg)
                    return pd.DataFrame({"Error": [error_msg]})
                
                # Convert to IDs and create tensors
                head_id = self.entity_to_id[head]
                tail_id = self.entity_to_id[tail]
                
                # Get device from model
                device = next(self.model.parameters()).device
                
                # Get correct predict_r signature based on model version
                try:
                    # Try older PyKEEN API version where predict_r takes a single batch argument
                    # Create batch of [head_id, tail_id] pairs
                    batch = torch.tensor([[head_id, tail_id]])
                    batch = batch.to(device)
                    predictions = self.model.predict_r(batch)
                except Exception as e:
                    logger.error(f"Error predicting relations: {e}")
                    return pd.DataFrame({"Error": [f"Failed to predict relations: {str(e)}"]})
                
                relation_scores = predictions[0].cpu().detach().numpy()
                
                # Validate relation indices are within bounds
                max_relation_id = len(self.id_to_relation) - 1
                
                # Get ranking information for top-k predictions
                ranking_info = self._get_ranking_info(relation_scores, k)
                
                # Convert to DataFrame with bounds checking
                for rank_info in ranking_info:
                    idx = rank_info['index']
                    if idx > max_relation_id or idx not in self.id_to_relation:
                        logger.warning(f"Skipping invalid relation index {idx} (max: {max_relation_id})")
                        continue
                        
                    relation_name = self.id_to_relation[idx]
                    results.append({
                        "Head": head,
                        "Relation": relation_name,
                        "Tail": tail,
                        "Raw_Score": rank_info['raw_score'],
                        "Rank": rank_info['rank'],
                        "Reciprocal_Rank": rank_info['reciprocal_rank'],
                        "Rank_Percentile": rank_info['rank_percentile'],
                        "Hits@1": rank_info['hits_at_1'],
                        "Hits@3": rank_info['hits_at_3'],
                        "Hits@5": rank_info['hits_at_5'],
                        "Hits@10": rank_info['hits_at_10']
                    })
                    
            # Predict tail entities
            elif head is not None and relation is not None and tail is None:
                # Validate inputs exist in training data
                if head not in self.entity_to_id:
                    error_msg = f"Entity '{head}' has ID {self.entity_to_id.get(head, 'unknown')} which exceeds model's entity vocabulary size {len(self.entity_to_id)}. This indicates a mismatch between training data and loaded model."
                    logger.error(error_msg)
                    return pd.DataFrame({"Error": [error_msg]})
                    
                if relation not in self.relation_to_id:
                    error_msg = f"Relation '{relation}' not in training data. Available relations: {list(self.relation_to_id.keys())[:10]}..."
                    logger.error(error_msg)
                    return pd.DataFrame({"Error": [error_msg]})
                
                # Convert to IDs and create tensors
                head_id = self.entity_to_id[head]
                relation_id = self.relation_to_id[relation]
                
                # Get device from model
                device = next(self.model.parameters()).device
                
                # Try both API versions
                try:
                    # Try batch version first (older API)
                    batch = torch.tensor([[head_id, relation_id]])
                    batch = batch.to(device)
                    predictions = self.model.predict_t(batch)
                except Exception as e:
                    logger.error(f"Error predicting tail entities: {e}")
                    return pd.DataFrame({"Error": [f"Failed to predict tail entities: {str(e)}"]})
                
                tail_scores = predictions[0].cpu().detach().numpy()
                
                # Validate entity indices are within bounds
                max_entity_id = len(self.id_to_entity) - 1
                
                # Get ranking information for top-k predictions
                ranking_info = self._get_ranking_info(tail_scores, k)
                
                # Convert to DataFrame with bounds checking
                for rank_info in ranking_info:
                    idx = rank_info['index']
                    if idx > max_entity_id or idx not in self.id_to_entity:
                        logger.warning(f"Skipping invalid entity index {idx} (max: {max_entity_id})")
                        continue
                        
                    entity = self.id_to_entity[idx]
                    results.append({
                        "Head": head,
                        "Relation": relation,
                        "Tail": entity,
                        "Raw_Score": rank_info['raw_score'],
                        "Rank": rank_info['rank'],
                        "Reciprocal_Rank": rank_info['reciprocal_rank'],
                        "Rank_Percentile": rank_info['rank_percentile'],
                        "Hits@1": rank_info['hits_at_1'],
                        "Hits@3": rank_info['hits_at_3'],
                        "Hits@5": rank_info['hits_at_5'],
                        "Hits@10": rank_info['hits_at_10']
                    })
            
            # Check if we have any valid results
            if not results:
                return pd.DataFrame({"Error": ["No valid predictions found. This may indicate a vocabulary mismatch between training and loaded model."]})
            
            # Create DataFrame with results
            df = pd.DataFrame(results)
            
            if not df.empty and "Rank" in df.columns:
                # Round numerical columns for better readability
                if "Reciprocal_Rank" in df.columns:
                    df["Reciprocal_Rank"] = df["Reciprocal_Rank"].round(4)
                if "Rank_Percentile" in df.columns:
                    df["Rank_Percentile"] = df["Rank_Percentile"].round(1)
                if "Raw_Score" in df.columns:
                    df["Raw_Score"] = df["Raw_Score"].round(4)
                
                # Data is already sorted by rank (best first)
                df = df.reset_index(drop=True)
                
            return df
            
        except Exception as e:
            logger.error(f"Error predicting triples: {e}")
            traceback.print_exc()
            return pd.DataFrame({"Error": [str(e)]})
    
    def batch_predict(self, prediction_type: str, entities: List[str], 
                     relation: Optional[str] = None, k: int = 5) -> pd.DataFrame:
        """
        Batch predict multiple entities with the same relation
        
        Args:
            prediction_type: Type of prediction ('head-batch', 'tail-batch')
            entities: List of entities (heads or tails)
            relation: The relation type to use for all predictions
            k: Number of top predictions to return per entity
            
        Returns:
            DataFrame with all predictions
        """
        try:
            if self.model is None:
                logger.error("No model loaded. Train or load a model first.")
                return pd.DataFrame({"Error": ["No model loaded"]})
                
            if prediction_type not in ['head-batch', 'tail-batch']:
                logger.error("prediction_type must be 'head-batch' or 'tail-batch'")
                return pd.DataFrame({"Error": ["prediction_type must be 'head-batch' or 'tail-batch'"]})
                
            if relation not in self.relation_to_id:
                return pd.DataFrame({"Error": [f"Relation '{relation}' not in training data"]})
                
            relation_id = self.relation_to_id[relation]
            
            # Filter valid entities that exist in the model
            valid_entities = [e for e in entities if e in self.entity_to_id]
            
            if not valid_entities:
                return pd.DataFrame({"Error": ["No valid entities found in the model"]})
                
            entity_ids = [self.entity_to_id[e] for e in valid_entities]
            
            # Prepare tensors
            entity_tensor = torch.tensor(entity_ids)
            relation_tensor = torch.tensor([relation_id] * len(valid_entities))
            
            # Move tensors to model device
            device = next(self.model.parameters()).device
            entity_tensor = entity_tensor.to(device)
            relation_tensor = relation_tensor.to(device)
            
            all_results = []
            
            # Head prediction
            if prediction_type == 'head-batch':
                # Predict head entities for each tail entity
                predictions = self.model.predict_h(entity_tensor, relation_tensor)
                head_scores = predictions.cpu().detach().numpy()
                
                for i, tail in enumerate(valid_entities):
                    # Get ranking information for this prediction
                    ranking_info = self._get_ranking_info(head_scores[i], k)
                    
                    for rank_info in ranking_info:
                        idx = rank_info['index']
                        head = self.id_to_entity[idx]
                        all_results.append({
                            "Head": head,
                            "Relation": relation,
                            "Tail": tail,
                            "Raw_Score": rank_info['raw_score'],
                            "Rank": rank_info['rank'],
                            "Reciprocal_Rank": rank_info['reciprocal_rank'],
                            "Rank_Percentile": rank_info['rank_percentile'],
                            "Hits@1": rank_info['hits_at_1'],
                            "Hits@3": rank_info['hits_at_3'],
                            "Hits@5": rank_info['hits_at_5'],
                            "Hits@10": rank_info['hits_at_10']
                        })
            
            # Tail prediction
            elif prediction_type == 'tail-batch':
                # Predict tail entities for each head entity
                predictions = self.model.predict_t(entity_tensor, relation_tensor)
                tail_scores = predictions.cpu().detach().numpy()
                
                for i, head in enumerate(valid_entities):
                    # Get ranking information for this prediction
                    ranking_info = self._get_ranking_info(tail_scores[i], k)
                    
                    for rank_info in ranking_info:
                        idx = rank_info['index']
                        tail = self.id_to_entity[idx]
                        all_results.append({
                            "Head": head,
                            "Relation": relation,
                            "Tail": tail,
                            "Raw_Score": rank_info['raw_score'],
                            "Rank": rank_info['rank'],
                            "Reciprocal_Rank": rank_info['reciprocal_rank'],
                            "Rank_Percentile": rank_info['rank_percentile'],
                            "Hits@1": rank_info['hits_at_1'],
                            "Hits@3": rank_info['hits_at_3'],
                            "Hits@5": rank_info['hits_at_5'],
                            "Hits@10": rank_info['hits_at_10']
                        })
            
            # Create and return result DataFrame
            results_df = pd.DataFrame(all_results)
            
            if not results_df.empty and "Rank" in results_df.columns:
                # Round numerical columns for better readability
                if "Reciprocal_Rank" in results_df.columns:
                    results_df["Reciprocal_Rank"] = results_df["Reciprocal_Rank"].round(4)
                if "Rank_Percentile" in results_df.columns:
                    results_df["Rank_Percentile"] = results_df["Rank_Percentile"].round(1)
                if "Raw_Score" in results_df.columns:
                    results_df["Raw_Score"] = results_df["Raw_Score"].round(4)
                
                # Sort by rank (best first)
                results_df = results_df.sort_values("Rank", ascending=True).reset_index(drop=True)
                
            return results_df
                
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            traceback.print_exc()
            return pd.DataFrame({"Error": [str(e)]})
    
    def find_similar_entities(self, entity: str, k: int = 10) -> pd.DataFrame:
        """
        Find semantically similar entities based on embedding distance
        
        Args:
            entity: The entity to find similar entities for
            k: Number of similar entities to return
            
        Returns:
            pd.DataFrame: DataFrame with similar entities
        """
        try:
            if self.model is None:
                logger.error("No model loaded. Train or load a model first.")
                return pd.DataFrame({"Error": ["No model loaded"]})
                
            if entity not in self.entity_to_id:
                logger.error(f"Entity '{entity}' not found in training data")
                return pd.DataFrame({"Error": [f"Entity '{entity}' not in training data"]})
                
            # Get entity ID and embedding
            entity_id = self.entity_to_id[entity]
            entity_tensor = torch.tensor([entity_id])
            
            # Get device from model
            device = next(self.model.parameters()).device
            entity_tensor = entity_tensor.to(device)
            
            # Get entity embedding
            entity_emb = self.model.entity_representations[0](indices=entity_tensor).cpu().detach().numpy()
            
            # Get all entity embeddings
            all_ids = torch.arange(len(self.entity_to_id))
            all_ids = all_ids.to(device)
            all_embs = self.model.entity_representations[0](indices=all_ids).cpu().detach().numpy()
            
            # Calculate Euclidean distance
            distances = np.linalg.norm(all_embs - entity_emb, axis=1)
            
            # Get top-k closest entities (excluding self)
            closest_indices = np.argsort(distances)
            closest_indices = closest_indices[1:k+1]  # Skip the first one (the entity itself)
            
            # Create result DataFrame
            results = []
            for idx in closest_indices:
                similar_entity = self.id_to_entity[idx]
                distance = float(distances[idx])
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity score
                
                results.append({
                    "Entity": entity,
                    "Similar Entity": similar_entity,
                    "Similarity": similarity,
                    "Distance": distance
                })
                
            df = pd.DataFrame(results)
            
            if not df.empty and "Distance" in df.columns:
                # Calculate ranking metrics for similarity
                distances = df["Distance"].values
                ranking_info = self._get_ranking_info(-distances, k)  # Negate because lower distance is better
                
                # Add ranking information
                for i, rank_info in enumerate(ranking_info):
                    if i < len(df):
                        df.loc[i, 'Rank'] = rank_info['rank']
                        df.loc[i, 'Reciprocal_Rank'] = rank_info['reciprocal_rank']
                        df.loc[i, 'Rank_Percentile'] = rank_info['rank_percentile']
                
                # Round for display
                df["Distance"] = df["Distance"].round(4)
                df["Similarity"] = df["Similarity"].round(4)
                if "Reciprocal_Rank" in df.columns:
                    df["Reciprocal_Rank"] = df["Reciprocal_Rank"].round(4)
                if "Rank_Percentile" in df.columns:
                    df["Rank_Percentile"] = df["Rank_Percentile"].round(1)
                
                # Data is already sorted by distance (best first)
                
            return df
            
        except Exception as e:
            logger.error(f"Error finding similar entities: {e}")
            traceback.print_exc()
            return pd.DataFrame({"Error": [str(e)]})
    
    def find_similar_movies(self, k: int = 10) -> pd.DataFrame:
        """
        Find similar movies based on all relationship patterns between movies
        
        Args:
            k: Number of similar movie recommendations to return
            
        Returns:
            DataFrame with similar movie recommendations
        """
        try:
            if self.model is None:
                logger.error("No model loaded. Train or load a model first.")
                return pd.DataFrame({"Error": ["No model loaded"]})
                
            # First, discover all relationship types between movies
            discover_relations_query = """
            MATCH (m1:Movie)-[r]->(m2:Movie)
            RETURN DISTINCT type(r) AS relation_type
            """
            relation_data = self.connector.execute_query(discover_relations_query)
            
            if not relation_data:
                logger.info("No direct movie-movie relationships found. Looking for indirect relationships...")
                # If no direct movie-movie relationships, look for movies connected through shared entities
                discover_relations_query = """
                MATCH (m1:Movie)-[r1]->(shared)<-[r2]-(m2:Movie)
                WHERE m1 <> m2
                RETURN DISTINCT type(r1) AS relation_type
                LIMIT 10
                """
                relation_data = self.connector.execute_query(discover_relations_query)
            
            if not relation_data:
                return pd.DataFrame({"Error": ["No relationships found between movies in the database"]})
            
            # Get all available movie relationship types
            available_relations = [row.get("relation_type") for row in relation_data if row.get("relation_type") is not None]
            
            # Filter to relations that exist in the trained model
            valid_relations = [rel for rel in available_relations if rel in self.relation_to_id]
            
            if not valid_relations:
                return pd.DataFrame({"Error": [f"No movie relationships from database found in trained model. Available in DB: {available_relations}, Available in model: {list(self.relation_to_id.keys())[:10]}..."]})
            
            logger.info(f"Using {len(valid_relations)} relationship types for movie similarity: {valid_relations}")
            
            # Extract Movie-type entities by checking against Neo4j
            query = """
            MATCH (m:Movie)
            RETURN m.title AS movie
            LIMIT 1000
            """
            movie_data = self.connector.execute_query(query)
            
            if not movie_data:
                return pd.DataFrame({"Error": ["No movies found in database"]})
                
            # Extract movie titles and filter to existing entities in model
            movies = [row.get("movie") for row in movie_data if row.get("movie") is not None]
            valid_movies = [m for m in movies if m in self.entity_to_id]
            
            if not valid_movies:
                return pd.DataFrame({"Error": ["No valid movie entities found in model"]})
            
            # Log some debugging info
            logger.info(f"Found {len(valid_movies)} valid movies out of {len(movies)} total")
            logger.info(f"Model entity dictionary has {len(self.entity_to_id)} entities")
            
            # Get device from model
            device = next(self.model.parameters()).device
            
            # Validate entity IDs against model's max entity ID
            max_entity_id = -1
            try:
                # Get max entity ID from the model's embedding matrix
                if hasattr(self.model, 'entity_representations') and len(self.model.entity_representations) > 0:
                    # Modern PyKEEN structure
                    max_entity_id = self.model.entity_representations[0]._embeddings.weight.shape[0] - 1
                elif hasattr(self.model, 'entity_embeddings'):
                    # Older PyKEEN structure
                    max_entity_id = self.model.entity_embeddings._embeddings.weight.shape[0] - 1
                else:
                    logger.warning("Could not determine max entity ID from model")
                
                # Ensure movie IDs are within range
                if max_entity_id > 0:
                    valid_movies = [m for m in valid_movies if self.entity_to_id[m] <= max_entity_id]
                    if not valid_movies:
                        return pd.DataFrame({"Error": ["No movies with valid entity IDs found"]})
                    logger.info(f"After range validation, {len(valid_movies)} movies remain")
            except Exception as e:
                logger.warning(f"Error determining max entity ID: {e}. Will continue with potential risk.")
            
            # Get movie embeddings
            movie_ids = [self.entity_to_id[m] for m in valid_movies]
            movie_tensor = torch.tensor(movie_ids)
            movie_tensor = movie_tensor.to(device)
            
            try:
                # Modern PyKEEN structure - get embeddings
                movie_embs = self.model.entity_representations[0](indices=movie_tensor).cpu().detach().numpy()
            except (IndexError, AttributeError) as e:
                try:
                    # Try older structure
                    movie_embs = self.model.entity_embeddings(movie_tensor).cpu().detach().numpy()
                except Exception as e2:
                    logger.error(f"Failed to get movie embeddings: {e2}")
                    return pd.DataFrame({"Error": [f"Failed to get movie embeddings: {str(e2)}"]})
            
            # Calculate movie similarity using multiple relationship patterns
            all_recommendations = []
            
            # For each movie, find similar movies based on all relationship patterns
            for i, source_movie in enumerate(valid_movies):
                similarity_scores = {}
                
                # For each valid relationship type, predict likely targets and accumulate scores
                for relation in valid_relations:
                    try:
                        # Get relation embedding
                        relation_id = self.relation_to_id[relation]
                        relation_tensor = torch.tensor([relation_id]).to(device)
                        
                        # Get relation embedding
                        try:
                            # Modern PyKEEN structure
                            relation_emb = self.model.relation_representations[0](indices=relation_tensor).cpu().detach().numpy()[0]
                        except (IndexError, AttributeError):
                            try:
                                # Try older structure
                                relation_emb = self.model.relation_embeddings(relation_tensor).cpu().detach().numpy()[0]
                            except Exception as e:
                                logger.warning(f"Failed to get relation embedding for {relation}: {e}")
                                continue
                        
                        # For each other movie, calculate how well it would fit as a target
                        for j, target_movie in enumerate(valid_movies):
                            if i == j:  # Skip self
                                continue
                                
                            # Calculate compatibility: how well does source + relation predict target?
                            # For TransE: h + r ≈ t, so we measure ||h + r - t||
                            if self.model_name == "TransE":
                                predicted = movie_embs[i] + relation_emb
                                distance = np.linalg.norm(predicted - movie_embs[j])
                                compatibility = 1.0 / (1.0 + distance)  # Convert distance to similarity
                            elif self.model_name in ["RotatE", "ComplEx"]:
                                # For other models, use direct embedding similarity as approximation
                                similarity = np.dot(movie_embs[i], movie_embs[j]) / (np.linalg.norm(movie_embs[i]) * np.linalg.norm(movie_embs[j]))
                                compatibility = (similarity + 1.0) / 2.0  # Normalize to [0, 1]
                            else:
                                # Default to TransE-style for other models
                                predicted = movie_embs[i] + relation_emb
                                distance = np.linalg.norm(predicted - movie_embs[j])
                                compatibility = 1.0 / (1.0 + distance)
                            
                            # Accumulate compatibility scores across all relations
                            if target_movie not in similarity_scores:
                                similarity_scores[target_movie] = 0.0
                            similarity_scores[target_movie] += compatibility
                    
                    except Exception as e:
                        logger.warning(f"Error processing relation {relation} for movie {source_movie}: {e}")
                        continue
                
                # Average the scores across all relations
                if similarity_scores:
                    for target_movie, total_score in similarity_scores.items():
                        avg_similarity = total_score / len(valid_relations)
                        all_recommendations.append({
                            "Source Movie": source_movie,
                            "Target Entity": target_movie,
                            "Similarity": avg_similarity,
                            "Relations Used": len(valid_relations)
                        })
            
            # Create result DataFrame
            results_df = pd.DataFrame(all_recommendations)
            
            if not results_df.empty:
                # Sort by similarity (highest first)
                results_df = results_df.sort_values("Similarity", ascending=False)
                
                # Round similarity scores for display
                results_df["Similarity"] = results_df["Similarity"].round(4)
                
                # Take top k overall
                results_df = results_df.head(k)
                
            return results_df
            
        except Exception as e:
            logger.error(f"Error in movie repurposing: {e}")
            traceback.print_exc()
            return pd.DataFrame({"Error": [str(e)]})
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate the current model
        
        Returns:
            Dict[str, float]: Dictionary with evaluation metrics
        """
        try:
            if self.model is None:
                logger.error("No model loaded. Train or load a model first.")
                return {"Error": "No model loaded"}
                
            if self.factory is None:
                if not self.prepare_training_data():
                    logger.error("Failed to prepare training data.")
                    return {"Error": "Failed to prepare training data"}
            
            # Split data into train/test if not done already
            if not hasattr(self, 'train_triples_factory') or not hasattr(self, 'test_triples_factory'):
                # Split data 80/20
                self.train_triples_factory, self.test_triples_factory = self.factory.split([0.8, 0.2])
            
            # Use PyKEEN's evaluation approach
            evaluator = RankBasedEvaluator()
            metrics = evaluator.evaluate(
                model=self.model,
                mapped_triples=self.test_triples_factory.mapped_triples,
                batch_size=128,
                additional_filter_triples=[self.train_triples_factory.mapped_triples],
            )
            
            # Extract key metrics
            results = {
                "hits_at_1": metrics.get_metric("hits_at_1"),
                "hits_at_3": metrics.get_metric("hits_at_3"),
                "hits_at_10": metrics.get_metric("hits_at_10"),
                "mean_rank": metrics.get_metric("mean_rank"),
                "mean_reciprocal_rank": metrics.get_metric("mean_reciprocal_rank")
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            traceback.print_exc()
            return {"Error": str(e)}
    
    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded model and its vocabulary
        
        Returns:
            Dict with model statistics
        """
        try:
            if self.model is None:
                return {"Error": "No model loaded"}
                
            stats = {
                "model_name": self.model_name or "Unknown",
                "entity_vocab_size": len(self.entity_to_id) if self.entity_to_id else 0,
                "relation_vocab_size": len(self.relation_to_id) if self.relation_to_id else 0,
                "total_triples": len(self.triples) if self.triples else 0,
            }
            
            # Try to get model's actual vocabulary sizes
            try:
                if hasattr(self.model, 'entity_representations') and len(self.model.entity_representations) > 0:
                    model_entity_count = self.model.entity_representations[0]._embeddings.weight.shape[0]
                    stats["model_entity_count"] = model_entity_count
                elif hasattr(self.model, 'entity_embeddings'):
                    model_entity_count = self.model.entity_embeddings._embeddings.weight.shape[0]
                    stats["model_entity_count"] = model_entity_count
                    
                if hasattr(self.model, 'relation_representations') and len(self.model.relation_representations) > 0:
                    model_relation_count = self.model.relation_representations[0]._embeddings.weight.shape[0]
                    stats["model_relation_count"] = model_relation_count
                elif hasattr(self.model, 'relation_embeddings'):
                    model_relation_count = self.model.relation_embeddings._embeddings.weight.shape[0]
                    stats["model_relation_count"] = model_relation_count
                    
            except Exception as e:
                stats["model_vocab_error"] = str(e)
                
            # Check for mismatches
            if "model_entity_count" in stats and stats["model_entity_count"] != stats["entity_vocab_size"]:
                stats["entity_mismatch"] = True
            if "model_relation_count" in stats and stats["model_relation_count"] != stats["relation_vocab_size"]:
                stats["relation_mismatch"] = True
                
            return stats
            
        except Exception as e:
            return {"Error": str(e)}
    
    def validate_model_mappings(self) -> bool:
        """
        Validate that the model's vocabulary matches the current mappings
        
        Returns:
            bool: True if consistent, False otherwise
        """
        try:
            if self.model is None:
                logger.error("No model loaded")
                return False
                
            # Get model statistics
            stats = self.get_model_stats()
            
            if "Error" in stats:
                logger.error(f"Failed to get model stats: {stats['Error']}")
                return False
                
            # Check for mismatches
            if stats.get("entity_mismatch", False):
                logger.warning(f"Entity count mismatch: Model has {stats.get('model_entity_count')} entities, mappings have {stats.get('entity_vocab_size')}")
                
            if stats.get("relation_mismatch", False):
                logger.warning(f"Relation count mismatch: Model has {stats.get('model_relation_count')} relations, mappings have {stats.get('relation_vocab_size')}")
                
            return not (stats.get("entity_mismatch", False) or stats.get("relation_mismatch", False))
            
        except Exception as e:
            logger.error(f"Error validating model mappings: {e}")
            return False
    
    def get_valid_entities_for_autocomplete(self, prefix: str = "", limit: int = 10) -> List[str]:
        """
        Get entities from the trained model for autocomplete functionality
        
        This ensures that only entities that were included in the training data
        are suggested, preventing the ID mismatch error.
        
        Args:
            prefix: The prefix to search for (if empty, returns all entities)
            limit: Maximum number of results to return
            
        Returns:
            List of matching entity names that are in the trained model
        """
        try:
            if not self.entity_to_id:
                logger.warning("No entity mappings available")
                return []
                
            # Get all entities from the training set
            all_entities = list(self.entity_to_id.keys())
            
            # Filter by prefix if provided
            if prefix:
                matching_entities = [e for e in all_entities if e.lower().startswith(prefix.lower())]
            else:
                matching_entities = all_entities
            
            # Sort alphabetically and limit results
            matching_entities.sort()
            return matching_entities[:limit]
            
        except Exception as e:
            logger.error(f"Error getting valid entities for autocomplete: {e}")
            return []
    
    def explain_ranking_system(self) -> str:
        """
        Provide a user-friendly explanation of the ranking metrics
        
        Returns:
            str: Explanation of how ranking metrics work
        """
        if self.model is None:
            return "No model loaded. Please train or load a model first."
            
        explanation = f"""
🎯 RANKING METRICS EXPLANATION

Your model ({self.model_name or 'Unknown'}) uses standard KG evaluation metrics.

📊 Ranking Metrics:
• Rank: Position of prediction among all candidates (1 = best)
• Reciprocal Rank: 1/rank (higher is better, max = 1.0)
• Rank Percentile: Better than X% of all candidates
• Hits@K: Binary indicator if prediction is in top-K

🔍 What You See:
• Raw_Score: Original model output
• Rank: 1, 2, 3, ... (lower is better)
• Reciprocal_Rank: 1.0, 0.5, 0.33, ... (higher is better)
• Rank_Percentile: 99.9%, 95.2%, ... (higher is better)
• Hits@1/3/5/10: 1 if in top-K, 0 otherwise

📈 Interpretation Guide:
• Rank 1-10: Excellent predictions (top 10)
• Rank 11-100: Good predictions (top 100)
• Rank 101-1000: Moderate predictions (top 1000)
• Rank >1000: Poor predictions

📊 Reciprocal Rank Guide:
• 1.0 (Rank 1): Perfect prediction
• 0.5 (Rank 2): Very good prediction
• 0.1 (Rank 10): Good prediction
• 0.01 (Rank 100): Moderate prediction
• <0.001: Poor prediction

🎯 Example: If "Megan Fox" ranks 15th out of 10,000 entities:
• Rank: 15
• Reciprocal Rank: 0.067
• Rank Percentile: 99.85%
• Hits@10: 0 (not in top 10)

These are standard metrics used in knowledge graph completion research.
"""
        return explanation
    
    def get_valid_relations_for_autocomplete(self, prefix: str = "", limit: int = 20) -> List[str]:
        """
        Get relations from the trained model for autocomplete functionality
        
        This ensures that only relations that were included in the training data
        are suggested, preventing the ID mismatch error.
        
        Args:
            prefix: The prefix to search for (if empty, returns all relations)
            limit: Maximum number of results to return
            
        Returns:
            List of matching relation names that are in the trained model
        """
        try:
            if not self.relation_to_id:
                logger.warning("No relation mappings available")
                return []
                
            # Get all relations from the training set
            all_relations = list(self.relation_to_id.keys())
            
            # Filter by prefix if provided
            if prefix:
                matching_relations = [r for r in all_relations if r.lower().startswith(prefix.lower())]
            else:
                matching_relations = all_relations
            
            # Sort alphabetically and limit results
            matching_relations.sort()
            return matching_relations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting valid relations for autocomplete: {e}")
            return []

# Helper function to get model class from name
def get_model_class(model_name: str):
    """
    Get PyKEEN model class from name
    """
    models = {
        "TransE": TransE,
        "PairRE": PairRE,
        "RotatE": RotatE, 
        "ComplEx": ComplEx
    }
    return models.get(model_name)

# Example usage
if __name__ == "__main__":
    # Create KG completion instance
    kg = PykeenMovieKG()
    
    # Load triples from a file (example)
    # kg.load_triples_from_file("data/knowledge_graph.tsv")
    
    # Split into train/test
    # kg.split_train_test(test_ratio=0.2)
    
    # Train a model
    # kg.train_model(model_name="TransE", epochs=100)
    
    # Make predictions
    # results = kg.predict_triplet(head="Entity1", relation="RelationA", tail=None, k=5)
    # print(results)
    
    # Evaluate model
    # metrics = kg.evaluate_model()
    # print(metrics) 