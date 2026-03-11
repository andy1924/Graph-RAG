"""Neo4j database connection and query management."""

import asyncio
from typing import List, Tuple, Dict, Any, Optional
from contextlib import asynccontextmanager

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jManager:
    """Manages Neo4j connections and queries."""
    
    def __init__(self, uri: str, user: str, password: str, database: str):
        """
        Initialize Neo4j manager.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Username for authentication
            password: Password for authentication
            database: Target database name
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None
        self._session = None
    
    async def initialize(self):
        """Initialize Neo4j connection."""
        try:
            from neo4j import AsyncGraphDatabase
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            
            # Test connection
            async with self._driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            
            logger.info(f"✓ Connected to Neo4j at {self.uri} database={self.database}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    async def close(self):
        """Close driver connection."""
        if self._driver:
            await self._driver.close()
            logger.info("✓ Closed Neo4j connection")
    
    @asynccontextmanager
    async def session(self):
        """Get async session context manager."""
        from neo4j import AsyncGraphDatabase
        if not self._driver:
            raise RuntimeError("Neo4jManager not initialized. Call initialize() first.")
        
        session = self._driver.session(database=self.database)
        try:
            yield session
        finally:
            await session.close()
    
    async def retrieve_context_for_question(
        self,
        question: str,
        entity_keywords: List[str],
        max_hops: int = 3,
        max_nodes: int = 50
    ) -> Tuple[str, List[str], List[Dict]]:
        """
        Retrieve context from graph for a question.
        
        Args:
            question: The user's question
            entity_keywords: Keywords to search for in entity names/descriptions
            max_hops: Maximum relationship depth to traverse
            max_nodes: Maximum number of nodes to return
        
        Returns:
            Tuple of (formatted_context, source_ids, raw_results)
        """
        try:
            async with self.session() as session:
                # Step 1: Find starting nodes with given keywords
                match_clause = " OR ".join(
                    [f"n.name CONTAINS '{kw}' OR n.description CONTAINS '{kw}'" 
                     for kw in entity_keywords if kw]
                )
                
                if not match_clause:
                    logger.warning("No keywords provided for Neo4j query")
                    return "", [], []
                
                query = f"""
                MATCH (n) 
                WHERE {match_clause}
                RETURN n.id as id, n.name as name, n.description as description, n.type as type
                LIMIT 10
                """
                
                result = await session.run(query)
                start_nodes = await result.values()
                
                if not start_nodes:
                    logger.info(f"No nodes found with keywords: {entity_keywords}")
                    return "", [], []
                
                logger.info(f"Found {len(start_nodes)} starting nodes")
                
                # Step 2: Traverse relationships up to max_hops
                start_ids = [n[0] for n in start_nodes]
                
                query = f"""
                MATCH path = (start)-[*1..{max_hops}]-(end)
                WHERE start.id IN {start_ids}
                WITH path, length(path) as depth
                UNWIND nodes(path) as node
                RETURN DISTINCT node.id as id, node.name as name, node.type as type, 
                        node.description as description, depth
                ORDER BY depth ASC
                LIMIT {max_nodes}
                """
                
                result = await session.run(query)
                results = await result.values()
                
                if not results:
                    logger.warning("Graph traversal returned no results")
                    return "", [], []
                
                logger.info(f"Retrieved {len(results)} nodes from graph")
                
                # Step 3: Format context
                context_lines = []
                sources = []
                
                for node_id, name, node_type, description, _ in results:
                    if name and description:
                        context_lines.append(f"{name} ({node_type}): {description}")
                        sources.append(node_id)
                
                formatted_context = "\n".join(context_lines)
                
                return formatted_context, list(set(sources)), results
        
        except Exception as e:
            logger.error(f"Error retrieving context from Neo4j: {str(e)}")
            return "", [], []
    
    async def get_node_properties(
        self,
        node_id: str
    ) -> Dict[str, Any]:
        """
        Get all properties of a node.
        
        Args:
            node_id: ID of the node to retrieve
        
        Returns:
            Dictionary of node properties
        """
        try:
            async with self.session() as session:
                query = "MATCH (n) WHERE n.id = $id RETURN properties(n) as props"
                result = await session.run(query, id=node_id)
                record = await result.single()
                
                if record:
                    return dict(record['props'])
                return {}
        
        except Exception as e:
            logger.error(f"Error retrieving node properties: {str(e)}")
            return {}
    
    async def get_relationships(
        self,
        node_id: str,
        relationship_type: Optional[str] = None,
        max_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get relationships connected to a node.
        
        Args:
            node_id: ID of the node
            relationship_type: Filter by relationship type (optional)
            max_depth: Maximum depth to traverse
        
        Returns:
            List of relationship dictionaries
        """
        try:
            async with self.session() as session:
                if relationship_type:
                    rel_clause = f":{relationship_type}"
                else:
                    rel_clause = ""
                
                query = f"""
                MATCH (n)-[r{rel_clause}*1..{max_depth}]-(m)
                WHERE n.id = $id
                RETURN r.type as type, m.name as target_name, m.id as target_id, r.weight as weight
                """
                
                result = await session.run(query, id=node_id)
                records = await result.values()
                
                relationships = []
                for rel_type, target_name, target_id, weight in records:
                    relationships.append({
                        "type": rel_type,
                        "target": target_name,
                        "target_id": target_id,
                        "weight": weight or 1.0
                    })
                
                return relationships
        
        except Exception as e:
            logger.error(f"Error retrieving relationships: {str(e)}")
            return []
