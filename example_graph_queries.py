"""
Example Cypher queries for analyzing the movie knowledge graph.

This file contains a collection of Cypher queries for performing 
exploratory analysis and subgraph extraction on the movie graph.
"""

# Basic exploratory queries
BASIC_EXPLORATORY_QUERIES = {
    "graph_size": """
        // Get the number of nodes and relationships in the graph
        MATCH (n)
        WITH count(n) AS nodeCount
        MATCH ()-[r]->()
        RETURN nodeCount, count(r) AS relationshipCount
    """,
    
    "node_counts_by_type": """
        // Count nodes by their label/type
        MATCH (n)
        RETURN labels(n)[0] AS nodeType, count(n) AS count
        ORDER BY count DESC
    """,
    
    "relationship_counts_by_type": """
        // Count relationships by their type
        MATCH ()-[r]->()
        RETURN type(r) AS relType, count(r) AS count
        ORDER BY count DESC
    """,
    
    "node_property_analysis": """
        // Analyze the properties of a specific node type
        MATCH (n:Movie)
        RETURN 
            min(n.release_year) AS earliest_movie,
            max(n.release_year) AS latest_movie,
            avg(n.vote_average) AS avg_rating,
            avg(n.runtime) AS avg_runtime
    """
}

# Degree distribution queries
DEGREE_DISTRIBUTION_QUERIES = {
    "actor_degree": """
        // Get the degree distribution for actors
        MATCH (p:Person)-[:ACTED_IN]->()
        WITH p, count(*) AS degree
        RETURN degree, count(p) AS actorCount
        ORDER BY degree
    """,
    
    "movie_degree": """
        // Get the in-degree distribution for movies (number of actors per movie)
        MATCH (m:Movie)<-[:ACTED_IN]-()
        WITH m, count(*) AS degree
        RETURN degree, count(m) AS movieCount
        ORDER BY degree
    """,
    
    "top_connected_people": """
        // Find people with the most connections
        MATCH (p:Person)
        WITH p, apoc.node.degree(p) AS degree
        RETURN p.name AS name, labels(p)[0] AS type, degree
        ORDER BY degree DESC
        LIMIT 20
    """,
    
    "director_actor_ratio": """
        // Calculate the ratio of actors to directors for each movie
        MATCH (m:Movie)<-[:ACTED_IN]-(a:Person)
        WITH m, count(a) AS actorCount
        MATCH (m)<-[:DIRECTED]-(d:Person)
        WITH m, actorCount, count(d) AS directorCount
        RETURN m.title AS movie, actorCount, directorCount, toFloat(actorCount)/directorCount AS ratio
        ORDER BY ratio DESC
        LIMIT 20
    """
}

# Centrality analysis queries
CENTRALITY_QUERIES = {
    "pagerank_setup": """
        // Create a graph projection for PageRank
        CALL gds.graph.project(
            'movieGraph',
            ['Movie', 'Person', 'Genre', 'Company', 'Country'],
            {
                ACTED_IN: {orientation: 'UNDIRECTED'},
                DIRECTED: {orientation: 'UNDIRECTED'},
                IN_GENRE: {orientation: 'UNDIRECTED'},
                PRODUCED: {orientation: 'UNDIRECTED'},
                SIMILAR_TO: {orientation: 'UNDIRECTED'},
                PRODUCED_IN: {orientation: 'UNDIRECTED'}
            }
        )
    """,
    
    "pagerank_people": """
        // Run PageRank and get influential people
        CALL gds.pageRank.stream('movieGraph')
        YIELD nodeId, score
        MATCH (p:Person) WHERE id(p) = nodeId
        RETURN p.name AS name, score AS pageRank
        ORDER BY pageRank DESC
        LIMIT 20
    """,
    
    "pagerank_movies": """
        // Run PageRank and get influential movies
        CALL gds.pageRank.stream('movieGraph')
        YIELD nodeId, score
        MATCH (m:Movie) WHERE id(m) = nodeId
        RETURN m.title AS title, score AS pageRank
        ORDER BY pageRank DESC
        LIMIT 20
    """,
    
    "betweenness_centrality": """
        // Run betweenness centrality
        CALL gds.betweenness.stream('movieGraph')
        YIELD nodeId, score
        MATCH (n) WHERE id(n) = nodeId
        RETURN labels(n)[0] AS type, n.name AS name, score AS betweenness
        ORDER BY betweenness DESC
        LIMIT 20
    """
}

# Community detection queries
COMMUNITY_DETECTION_QUERIES = {
    "louvain_communities": """
        // Detect communities using Louvain algorithm
        CALL gds.louvain.stream('movieGraph')
        YIELD nodeId, communityId
        MATCH (n) WHERE id(n) = nodeId
        RETURN communityId, count(*) AS communitySize
        ORDER BY communitySize DESC
        LIMIT 10
    """,
    
    "louvain_community_composition": """
        // Get the composition of the top communities
        CALL gds.louvain.stream('movieGraph')
        YIELD nodeId, communityId
        MATCH (n) WHERE id(n) = nodeId
        WITH communityId, labels(n)[0] AS nodeType, count(*) AS typeCount
        ORDER BY communityId, typeCount DESC
        RETURN communityId, collect({type: nodeType, count: typeCount}) AS composition
        LIMIT 5
    """,
    
    "community_examples": """
        // Get examples of nodes from a specific community
        CALL gds.louvain.stream('movieGraph')
        YIELD nodeId, communityId
        WITH nodeId, communityId WHERE communityId = $communityId
        MATCH (n) WHERE id(n) = nodeId
        RETURN labels(n)[0] AS type, n.name AS name, n.title AS title
        LIMIT 20
    """
}

# Subgraph extraction queries
SUBGRAPH_QUERIES = {
    "actor_network": """
        // Extract a subgraph of an actor's collaborations
        MATCH (a:Person {name: $actorName})-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(costar:Person)
        RETURN a, m, costar, 
               (a)-[:ACTED_IN]->(m) AS a_acted,
               (costar)-[:ACTED_IN]->(m) AS costar_acted
        LIMIT 100
    """,
    
    "director_actor_collaborations": """
        // Extract a director's collaboration network
        MATCH (d:Person {name: $directorName})-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person)
        RETURN d, m, a, 
               (d)-[:DIRECTED]->(m) AS directed,
               (a)-[:ACTED_IN]->(m) AS acted_in
        LIMIT 100
    """,
    
    "genre_movie_network": """
        // Extract a genre-based movie network
        MATCH (g:Genre {name: $genreName})<-[:IN_GENRE]-(m:Movie)
        WHERE m.vote_average >= 7.0
        WITH g, m LIMIT 30
        MATCH (m)-[:IN_GENRE]->(og:Genre)
        OPTIONAL MATCH (p:Person)-[:DIRECTED]->(m)
        RETURN g, m, og, p, 
               (m)-[:IN_GENRE]->(g) AS main_genre,
               (m)-[:IN_GENRE]->(og) AS other_genre,
               (p)-[:DIRECTED]->(m) AS directed
        LIMIT 200
    """,
    
    "similar_movie_network": """
        // Extract a network of similar movies
        MATCH (m:Movie {title: $movieTitle})-[s:SIMILAR_TO]-(sm:Movie)
        WITH m, sm, s
        MATCH (m)-[:IN_GENRE]->(g:Genre)<-[:IN_GENRE]-(sm)
        RETURN m, sm, g, s, 
               (m)-[:IN_GENRE]->(g) AS m_genre,
               (sm)-[:IN_GENRE]->(g) AS sm_genre
        LIMIT 100
    """,
    
    "temporal_evolution": """
        // Extract a subgraph showing evolution of a genre over time
        MATCH (g:Genre {name: $genreName})<-[:IN_GENRE]-(m:Movie)
        WHERE m.release_year >= $startYear AND m.release_year <= $endYear
        WITH g, m
        ORDER BY m.vote_average * m.vote_count DESC
        LIMIT 50
        MATCH (m)-[:IN_GENRE]->(og:Genre)
        OPTIONAL MATCH (d:Person)-[:DIRECTED]->(m)
        RETURN g, m, og, d,
               (m)-[:IN_GENRE]->(g) AS in_main_genre,
               (m)-[:IN_GENRE]->(og) AS in_any_genre,
               (d)-[:DIRECTED]->(m) AS directed
        LIMIT 200
    """
}

# Path analysis and reachability queries
PATH_ANALYSIS_QUERIES = {
    "shortest_path_actors": """
        // Find shortest paths between actors (e.g., "Six Degrees of Kevin Bacon")
        MATCH path = shortestPath(
          (p1:Person {name: $actor1})-[:ACTED_IN*]-(p2:Person {name: $actor2})
        )
        RETURN path
    """,
    
    "actor_director_paths": """
        // Find all paths between an actor and director
        MATCH path = (a:Person {name: $actorName})-[:ACTED_IN|DIRECTED*..4]-(d:Person {name: $directorName})
        RETURN path
        LIMIT 10
    """,
    
    "movie_genre_transition": """
        // Find how genres connect through movies
        MATCH path = shortestPath((g1:Genre {name: $genre1})-[:IN_GENRE*]-(g2:Genre {name: $genre2}))
        RETURN path
    """
}

# Advanced analysis queries
ADVANCED_ANALYSIS_QUERIES = {
    "genre_correlation": """
        // Find correlations between genres
        MATCH (g1:Genre)<-[:IN_GENRE]-(m:Movie)-[:IN_GENRE]->(g2:Genre)
        WHERE g1.name < g2.name
        WITH g1.name AS genre1, g2.name AS genre2, count(m) AS movieCount
        RETURN genre1, genre2, movieCount
        ORDER BY movieCount DESC
        LIMIT 20
    """,
    
    "actor_type_analysis": """
        // Analyze which actors tend to be in which types of genres
        MATCH (a:Person)-[:ACTED_IN]->(m:Movie)-[:IN_GENRE]->(g:Genre)
        WITH a, g.name AS genre, count(m) AS movieCount
        WHERE movieCount >= 3
        RETURN a.name AS actor, collect({genre: genre, count: movieCount}) AS genreDistribution
        ORDER BY size(collect({genre: genre, count: movieCount})) DESC
        LIMIT 20
    """,
    
    "director_success": """
        // Analyze directors by the average ratings of their movies
        MATCH (d:Person)-[:DIRECTED]->(m:Movie)
        WHERE m.vote_count > 100
        WITH d, avg(m.vote_average) AS avgRating, count(m) AS movieCount
        WHERE movieCount >= 3
        RETURN d.name AS director, avgRating, movieCount
        ORDER BY avgRating DESC
        LIMIT 20
    """,
    
    "genre_popularity_over_time": """
        // Analyze how genre popularity has changed over time
        MATCH (g:Genre)<-[:IN_GENRE]-(m:Movie)
        WHERE m.release_year >= 1950 AND m.release_year <= 2020
        WITH g.name AS genre, m.release_year/10*10 AS decade, count(m) AS movieCount
        RETURN genre, decade, movieCount
        ORDER BY genre, decade
    """
}

def get_query(category, name):
    """
    Get a specific query by category and name.
    
    Args:
        category: The category of queries
        name: The name of the specific query
        
    Returns:
        str: The Cypher query
    """
    categories = {
        'basic': BASIC_EXPLORATORY_QUERIES,
        'degree': DEGREE_DISTRIBUTION_QUERIES,
        'centrality': CENTRALITY_QUERIES,
        'community': COMMUNITY_DETECTION_QUERIES,
        'subgraph': SUBGRAPH_QUERIES,
        'path': PATH_ANALYSIS_QUERIES,
        'advanced': ADVANCED_ANALYSIS_QUERIES
    }
    
    if category.lower() not in categories:
        raise ValueError(f"Category '{category}' not found. Available categories: {list(categories.keys())}")
    
    query_dict = categories[category.lower()]
    if name not in query_dict:
        raise ValueError(f"Query '{name}' not found in category '{category}'. Available queries: {list(query_dict.keys())}")
    
    return query_dict[name]

def print_available_queries():
    """Print all available queries by category."""
    categories = {
        'Basic Exploratory Queries': BASIC_EXPLORATORY_QUERIES,
        'Degree Distribution Queries': DEGREE_DISTRIBUTION_QUERIES,
        'Centrality Analysis Queries': CENTRALITY_QUERIES,
        'Community Detection Queries': COMMUNITY_DETECTION_QUERIES,
        'Subgraph Extraction Queries': SUBGRAPH_QUERIES,
        'Path Analysis Queries': PATH_ANALYSIS_QUERIES,
        'Advanced Analysis Queries': ADVANCED_ANALYSIS_QUERIES
    }
    
    print("Available Cypher Queries:")
    print("=========================")
    
    for category_name, queries in categories.items():
        print(f"\n{category_name}:")
        for query_name in queries.keys():
            print(f"  - {query_name}")
    
    print("\nUse get_query(category, name) to get a specific query.")
    print("Example: get_query('subgraph', 'actor_network')")

if __name__ == "__main__":
    print_available_queries() 