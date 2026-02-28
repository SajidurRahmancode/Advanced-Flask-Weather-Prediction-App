"""
Advanced RAG Service with Multi-Query Retrieval
Enhanced retrieval with query generation, reranking, and hybrid search
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

# Vector store and embeddings
try:
    from langchain_community.vectorstores import Chroma
    from langchain.schema import Document
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

# Local embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImprovedLocalEmbeddings:
    """Enhanced local embeddings using better model"""
    
    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        """Initialize with improved embedding model
        
        Args:
            model_name: Options:
                - all-MiniLM-L12-v2 (384 dim, better quality than L6)
                - all-mpnet-base-v2 (768 dim, highest quality if RAM allows)
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers not available")
        
        try:
            logger.info(f"🔄 Loading improved embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Loaded {model_name} (dimension: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query text"""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"❌ Query embedding failed: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        try:
            embeddings = self.model.encode(
                texts, 
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 10
            )
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(f"❌ Document embedding failed: {e}")
            raise


class AdvancedRAGService:
    """Advanced RAG with multi-query retrieval, reranking, and hybrid search"""
    
    def __init__(
        self, 
        vector_db_path: str,
        lm_studio_service=None,
        embedding_model: str = "all-MiniLM-L12-v2"
    ):
        """Initialize advanced RAG service
        
        Args:
            vector_db_path: Path to ChromaDB vector store
            lm_studio_service: LM Studio service for query generation
            embedding_model: Embedding model to use
        """
        self.vector_db_path = vector_db_path
        self.lm_studio_service = lm_studio_service
        self.available = False
        
        logger.info("🚀 Initializing Advanced RAG Service...")
        
        # Initialize improved embeddings
        try:
            self.embeddings = ImprovedLocalEmbeddings(embedding_model)
            logger.info(f"✅ Using {embedding_model} embeddings")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {e}")
            return
        
        # Initialize vector store
        if not CHROMA_AVAILABLE:
            logger.error("❌ ChromaDB not available")
            return
        
        try:
            self.vectorstore = Chroma(
                persist_directory=vector_db_path,
                embedding_function=self.embeddings,
                collection_name="weather_patterns_advanced"
            )
            logger.info("✅ ChromaDB vector store loaded")
            self.available = True
        except Exception as e:
            logger.error(f"❌ Failed to load vector store: {e}")
            self.available = False
    
    async def multi_query_retrieval(
        self, 
        location: str, 
        days: int,
        season: Optional[str] = None,
        k: int = 10
    ) -> Dict[str, Any]:
        """Perform multi-query retrieval for comprehensive context
        
        Args:
            location: Location for weather prediction
            days: Number of days to predict
            season: Optional season filter
            k: Number of documents to retrieve
        
        Returns:
            Dict with retrieved documents and metadata
        """
        if not self.available:
            logger.error("❌ Advanced RAG service not available")
            return {"documents": [], "query_variations": [], "total_retrieved": 0}
        
        logger.info(f"🔍 Multi-query retrieval for {location} ({days} days)")
        
        # Generate query variations
        query_variations = await self._generate_query_variations(location, days, season)
        logger.info(f"📝 Generated {len(query_variations)} query variations")
        
        # Retrieve documents for each query variation
        all_results = []
        seen_content = set()
        
        for i, query in enumerate(query_variations):
            try:
                # Perform similarity search
                filter_dict = self._create_filter(location, season)
                results = self.vectorstore.similarity_search(
                    query,
                    k=max(5, k // len(query_variations)),
                    filter=filter_dict
                )
                
                # Deduplicate based on content
                for doc in results:
                    # Use first 100 characters as unique identifier
                    content_id = doc.page_content[:100]
                    if content_id not in seen_content:
                        all_results.append(doc)
                        seen_content.add(content_id)
                
                logger.info(f"  Query {i+1}: Retrieved {len(results)} documents")
                
            except Exception as e:
                logger.warning(f"⚠️ Query {i+1} failed: {e}")
                continue
        
        logger.info(f"📊 Total unique documents: {len(all_results)}")
        
        # Rerank results
        reranked_results = self._rerank_local(query_variations[0], all_results)
        
        # Return top k
        final_results = reranked_results[:k]
        
        return {
            "documents": final_results,
            "query_variations": query_variations,
            "total_retrieved": len(all_results),
            "after_deduplication": len(reranked_results),
            "final_count": len(final_results)
        }
    
    async def _generate_query_variations(
        self, 
        location: str, 
        days: int,
        season: Optional[str]
    ) -> List[str]:
        """Generate multiple query variations for comprehensive retrieval
        
        Uses LM Studio (Qwen3-14B) to generate diverse search queries
        """
        # Base query (always included)
        base_query = f"weather patterns for {location}"
        
        # If LM Studio not available, use predefined variations
        if not self.lm_studio_service or not self.lm_studio_service.available:
            logger.warning("⚠️ LM Studio not available, using predefined variations")
            return self._get_predefined_variations(location, days, season)
        
        # Generate variations using Qwen3-14B
        prompt = f"""Generate 5 different search queries to find relevant historical weather patterns for:
- Location: {location}
- Forecast period: {days} days
- Season: {season or "current"}

Each query should focus on a different aspect:
1. Direct weather pattern match
2. Seasonal patterns and trends
3. Temperature and humidity patterns
4. Precipitation and wind patterns  
5. Historical weather extremes or anomalies

Output format (one query per line):
1. [query text]
2. [query text]
3. [query text]
4. [query text]
5. [query text]

Be specific and use meteorological terms."""

        try:
            response = await asyncio.to_thread(
                self.lm_studio_service.generate_text,
                prompt,
                max_tokens=300,
                temperature=0.7  # Higher for diversity
            )
            
            if response:
                queries = self._parse_queries(response)
                if len(queries) >= 3:
                    logger.info(f"✅ Generated {len(queries)} queries using Qwen3-14B")
                    return [base_query] + queries
                    
        except Exception as e:
            logger.warning(f"⚠️ Query generation failed: {e}")
        
        # Fallback to predefined
        return self._get_predefined_variations(location, days, season)
    
    def _get_predefined_variations(
        self, 
        location: str, 
        days: int,
        season: Optional[str]
    ) -> List[str]:
        """Get predefined query variations as fallback"""
        season_str = season or "all seasons"
        
        return [
            f"weather patterns for {location}",
            f"{season_str} weather in {location}",
            f"temperature and humidity trends {location}",
            f"precipitation patterns {location}",
            f"historical weather extremes {location}",
            f"{days}-day forecast patterns {location}"
        ]
    
    def _parse_queries(self, text: str) -> List[str]:
        """Extract queries from LLM response"""
        import re
        
        queries = []
        lines = text.split('\n')
        
        for line in lines:
            # Match patterns like "1. query" or "1) query" or "Query 1: query"
            match = re.search(r'(?:^\d+[\.)]\s*|^Query\s*\d+:\s*)(.+)', line.strip())
            if match:
                query = match.group(1).strip()
                if query and len(query) > 10:  # Valid query
                    queries.append(query)
        
        return queries[:5]  # Return up to 5 queries
    
    def _rerank_local(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using local embedding similarity
        
        Args:
            query: Original query
            documents: Documents to rerank
        
        Returns:
            Reranked documents (most relevant first)
        """
        if not documents:
            return []
        
        try:
            # Get query embedding
            query_embedding = np.array(self.embeddings.embed_query(query))
            
            # Calculate similarity scores
            scored_docs = []
            for doc in documents:
                # Get document embedding
                doc_embedding = np.array(self.embeddings.embed_query(doc.page_content[:500]))
                
                # Cosine similarity (embeddings are normalized)
                similarity = np.dot(query_embedding, doc_embedding)
                
                scored_docs.append((doc, float(similarity)))
            
            # Sort by similarity (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"🎯 Reranked {len(documents)} documents")
            
            # Return only documents (drop scores)
            return [doc for doc, score in scored_docs]
            
        except Exception as e:
            logger.warning(f"⚠️ Reranking failed: {e}, returning original order")
            return documents
    
    def _create_filter(self, location: str, season: Optional[str]) -> Optional[Dict]:
        """Create metadata filter for more relevant retrieval"""
        filter_dict = {}
        
        if location:
            filter_dict["location"] = location
        
        if season:
            filter_dict["season"] = season
        
        return filter_dict if filter_dict else None
    
    async def contextual_compression(
        self, 
        query: str, 
        documents: List[Document],
        max_docs: int = 5
    ) -> List[Document]:
        """Use LLM to compress and select most relevant passages
        
        Args:
            query: Search query
            documents: Documents to filter
            max_docs: Maximum documents to return
        
        Returns:
            Filtered relevant documents
        """
        if not self.lm_studio_service or not self.lm_studio_service.available:
            logger.warning("⚠️ LM Studio not available for compression, returning top documents")
            return documents[:max_docs]
        
        relevant_docs = []
        
        for doc in documents[:max_docs * 2]:  # Check 2x to get best N
            # Ask LLM if document is relevant
            prompt = f"""Is this weather data relevant for the query?

Query: {query}

Weather Data (excerpt):
{doc.page_content[:500]}

Answer only 'yes' or 'no' with a brief reason (one sentence)."""

            try:
                response = await asyncio.to_thread(
                    self.lm_studio_service.generate_text,
                    prompt,
                    max_tokens=50,
                    temperature=0.1
                )
                
                if response and 'yes' in response.lower():
                    relevant_docs.append(doc)
                    
                    if len(relevant_docs) >= max_docs:
                        break
                        
            except Exception as e:
                logger.warning(f"⚠️ Compression check failed: {e}")
                continue
        
        logger.info(f"🎯 Compressed to {len(relevant_docs)} relevant documents")
        
        return relevant_docs if relevant_docs else documents[:max_docs]
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "service": "Advanced Multi-Query RAG",
            "available": self.available,
            "embedding_model": self.embeddings.model_name if self.available else None,
            "embedding_dimension": self.embeddings.embedding_dim if self.available else None,
            "vector_db_path": self.vector_db_path,
            "features": {
                "multi_query_retrieval": True,
                "local_reranking": True,
                "contextual_compression": self.lm_studio_service is not None,
                "llm_query_generation": self.lm_studio_service is not None
            }
        }


# Test function
async def test_advanced_rag():
    """Test advanced RAG service"""
    print("🧪 Testing Advanced RAG Service...")
    
    # Initialize (you'll need to provide actual paths)
    vector_db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_db')
    
    try:
        from backend.lmstudio_service import get_lm_studio_service
        lm_studio = get_lm_studio_service()
    except:
        lm_studio = None
        print("⚠️ LM Studio not available for testing")
    
    service = AdvancedRAGService(
        vector_db_path=vector_db_path,
        lm_studio_service=lm_studio,
        embedding_model="all-MiniLM-L12-v2"
    )
    
    if not service.available:
        print("❌ Service not available")
        return
    
    # Test retrieval
    print("\n🔍 Testing multi-query retrieval...")
    results = await service.multi_query_retrieval(
        location="Tokyo",
        days=3,
        k=5
    )
    
    print(f"\n📊 Results:")
    print(f"  Query variations: {len(results['query_variations'])}")
    print(f"  Total retrieved: {results['total_retrieved']}")
    print(f"  Final count: {results['final_count']}")
    
    print(f"\n📝 Query Variations:")
    for i, query in enumerate(results['query_variations'], 1):
        print(f"  {i}. {query}")
    
    print(f"\n📄 Top Documents:")
    for i, doc in enumerate(results['documents'][:3], 1):
        print(f"  {i}. {doc.page_content[:100]}...")
    
    # Get stats
    stats = service.get_service_stats()
    print(f"\n📊 Service Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Run async test
    asyncio.run(test_advanced_rag())
