"""
Retrieval Evaluation Module for Assessing Retrieval Quality
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from langchain_classic.schema import Document


class RetrievalEvaluator:
    """Evaluate retrieval quality with various metrics"""
    
    def __init__(self):
        self.query_history = []
        self.retrieval_history = []
    
    def calculate_relevance_scores(
        self,
        results: List[Tuple[Document, float]]
    ) -> List[Dict[str, Any]]:
        """Calculate and format relevance scores for retrieved documents"""
        
        scored_results = []
        
        for idx, (doc, score) in enumerate(results):
            scored_results.append({
                'rank': idx + 1,
                'score': float(score),
                'normalized_score': self._normalize_score(score, results),
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_tier': self._get_relevance_tier(idx, len(results))
            })
        
        return scored_results
    
    def _normalize_score(
        self,
        score: float,
        all_results: List[Tuple[Document, float]]
    ) -> float:
        """Normalize score to 0-1 range based on min-max in result set"""
        
        if not all_results:
            return 0.0
        
        scores = [s for _, s in all_results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return 1.0
        
        # Invert if needed (lower scores are better for L2 distance)
        normalized = (score - min_score) / (max_score - min_score)
        return normalized
    
    def _get_relevance_tier(self, rank: int, total: int) -> str:
        """Categorize relevance based on rank"""
        
        percentile = (rank + 1) / total
        
        if percentile <= 0.2:
            return "Highly Relevant"
        elif percentile <= 0.5:
            return "Relevant"
        elif percentile <= 0.8:
            return "Somewhat Relevant"
        else:
            return "Less Relevant"
    
    def calculate_precision_at_k(
        self,
        results: List[Tuple[Document, float]],
        relevant_doc_ids: List[str],
        k: int = None
    ) -> float:
        """
        Calculate Precision@K
        
        Precision@K = (# of relevant docs in top K) / K
        """
        
        if k is None:
            k = len(results)
        
        results_k = results[:k]
        
        relevant_count = 0
        for doc, _ in results_k:
            doc_id = doc.metadata.get('source_pdf_unique_id', '')
            if doc_id in relevant_doc_ids:
                relevant_count += 1
        
        return relevant_count / k if k > 0 else 0.0
    
    def calculate_recall_at_k(
        self,
        results: List[Tuple[Document, float]],
        relevant_doc_ids: List[str],
        k: int = None
    ) -> float:
        """
        Calculate Recall@K
        
        Recall@K = (# of relevant docs in top K) / (total # of relevant docs)
        """
        
        if not relevant_doc_ids:
            return 0.0
        
        if k is None:
            k = len(results)
        
        results_k = results[:k]
        
        relevant_count = 0
        for doc, _ in results_k:
            doc_id = doc.metadata.get('source_pdf_unique_id', '')
            if doc_id in relevant_doc_ids:
                relevant_count += 1
        
        return relevant_count / len(relevant_doc_ids)
    
    def calculate_mrr(
        self,
        results: List[Tuple[Document, float]],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        MRR = 1 / rank of first relevant document
        """
        
        for idx, (doc, _) in enumerate(results, start=1):
            doc_id = doc.metadata.get('source_pdf_unique_id', '')
            if doc_id in relevant_doc_ids:
                return 1.0 / idx
        
        return 0.0
    
    def calculate_ndcg_at_k(
        self,
        results: List[Tuple[Document, float]],
        relevance_scores: Dict[str, float],
        k: int = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K)
        
        NDCG@K measures ranking quality considering position
        """
        
        if k is None:
            k = len(results)
        
        results_k = results[:k]
        
        # Calculate DCG
        dcg = 0.0
        for idx, (doc, _) in enumerate(results_k, start=1):
            doc_id = doc.metadata.get('source_pdf_unique_id', '')
            relevance = relevance_scores.get(doc_id, 0.0)
            dcg += relevance / np.log2(idx + 1)
        
        # Calculate IDCG (ideal DCG with perfect ranking)
        sorted_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for idx, relevance in enumerate(sorted_relevances, start=1):
            idcg += relevance / np.log2(idx + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_diversity(
        self,
        results: List[Tuple[Document, float]]
    ) -> Dict[str, Any]:
        """Calculate diversity metrics for retrieved documents"""
        
        # Document type diversity
        chunk_types = [doc.metadata.get('chunk_type', 'unknown') for doc, _ in results]
        type_counts = {}
        for chunk_type in chunk_types:
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        # Source diversity
        sources = [doc.metadata.get('pdf_name', 'unknown') for doc, _ in results]
        unique_sources = len(set(sources))
        
        # Page diversity
        pages = [doc.metadata.get('source_page', 0) for doc, _ in results]
        unique_pages = len(set(pages))
        
        return {
            'chunk_type_distribution': type_counts,
            'unique_sources': unique_sources,
            'unique_pages': unique_pages,
            'diversity_score': unique_sources / len(results) if results else 0.0
        }
    
    def log_query(
        self,
        query: str,
        results: List[Tuple[Document, float]],
        config: Dict[str, Any]
    ) -> None:
        """Log query and results for analysis"""
        
        self.query_history.append({
            'query': query,
            'num_results': len(results),
            'config': config,
            'top_score': results[0][1] if results else 0.0
        })
        
        self.retrieval_history.append({
            'query': query,
            'results': self.calculate_relevance_scores(results)
        })
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get statistics across all queries"""
        
        if not self.query_history:
            return {
                'total_queries': 0,
                'avg_results': 0,
                'avg_top_score': 0
            }
        
        total_queries = len(self.query_history)
        avg_results = np.mean([q['num_results'] for q in self.query_history])
        avg_top_score = np.mean([q['top_score'] for q in self.query_history])
        
        return {
            'total_queries': total_queries,
            'avg_results': avg_results,
            'avg_top_score': avg_top_score,
            'queries': self.query_history
        }
    
    def compare_retrievals(
        self,
        results_a: List[Tuple[Document, float]],
        results_b: List[Tuple[Document, float]],
        config_a: Dict[str, Any],
        config_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two different retrieval configurations"""
        
        # Get top documents from each
        docs_a = set([doc.metadata.get('chunk_index', idx) for idx, (doc, _) in enumerate(results_a)])
        docs_b = set([doc.metadata.get('chunk_index', idx) for idx, (doc, _) in enumerate(results_b)])
        
        # Calculate overlap
        intersection = docs_a.intersection(docs_b)
        union = docs_a.union(docs_b)
        
        overlap_ratio = len(intersection) / len(union) if union else 0.0
        
        # Calculate score differences
        scores_a = [score for _, score in results_a]
        scores_b = [score for _, score in results_b]
        
        return {
            'overlap_ratio': overlap_ratio,
            'unique_to_a': len(docs_a - docs_b),
            'unique_to_b': len(docs_b - docs_a),
            'avg_score_a': np.mean(scores_a) if scores_a else 0.0,
            'avg_score_b': np.mean(scores_b) if scores_b else 0.0,
            'config_a': config_a,
            'config_b': config_b
        }
    
    def export_results(self) -> Dict[str, Any]:
        """Export all evaluation results"""
        
        return {
            'query_history': self.query_history,
            'retrieval_history': self.retrieval_history,
            'statistics': self.get_query_statistics()
        }
