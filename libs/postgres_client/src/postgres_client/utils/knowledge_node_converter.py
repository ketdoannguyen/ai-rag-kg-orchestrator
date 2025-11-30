"""
Converter utilities for KnowledgeNode between Pydantic models and ORM classes.

NOTE: This module ONLY handles KnowledgeNode conversion (which is complex due to multiple subtypes).
Other Pydantic models (User, DataSource, EmbeddingStructure, etc.) have their own
to_orm() and from_orm() methods defined in their respective classes.

When KnowledgeNode Pydantic models change, update the conversion logic here.
"""

from __future__ import annotations

from typing import Any, Type

from ..models.knowledge import KnowledgeNode as KnowledgeNodePydantic
from ..models.detail_nodes import ImageNode, ParagraphNode, TableNode
from ..models.structure_nodes import CorpusNode, StructureNode, SubtopicNode, TopicNode
from ..orm_models.knowledge import KnowledgeNodeORM


def pydantic_to_orm(pydantic_model: KnowledgeNodePydantic) -> KnowledgeNodeORM:
    """
    Convert KnowledgeNode Pydantic model to ORM instance.
    
    NOTE: This function ONLY works for KnowledgeNode and its subtypes.
    For other models (User, DataSource, etc.), use their to_orm() method instead.
    
    This is the main conversion function for KnowledgeNode. When Pydantic models change,
    update this function to reflect the changes.
    
    Args:
        pydantic_model: Validated KnowledgeNode Pydantic model instance
        
    Returns:
        KnowledgeNodeORM instance ready to be saved to database
        
    Example:
        >>> paragraph = ParagraphNode(...)
        >>> orm_node = pydantic_to_orm(paragraph)
    """
    # Convert Pydantic model to dict, excluding children (handled by relationships)
    data = pydantic_model.model_dump(exclude={'children'})
    
    # Convert attributes and metadata to dict if they're Pydantic models
    attributes = data.get('attributes', {})
    if hasattr(attributes, 'model_dump'):
        attributes = attributes.model_dump()
    
    metadata = data.get('metadata', {})
    if hasattr(metadata, 'model_dump'):
        metadata = metadata.model_dump()
    
    # Create ORM instance
    orm_instance = KnowledgeNodeORM(
        id=data.get('id') or pydantic_model.id,
        source_id=data.get('source_id') or pydantic_model.source_id,
        parent_id=data.get('parent_id') or pydantic_model.parent_id,
        semantic_level=pydantic_model.semantic_level,
        physical_level=pydantic_model.physical_level,
        attributes=attributes,
        node_metadata=metadata,  # Use node_metadata (ORM attribute name)
    )
    
    return orm_instance


def orm_to_pydantic(
    orm_instance: KnowledgeNodeORM,
    pydantic_class: Type[KnowledgeNodePydantic] | None = None
) -> KnowledgeNodePydantic:
    """
    Convert KnowledgeNode ORM instance to Pydantic model.
    
    NOTE: This function ONLY works for KnowledgeNodeORM.
    For other ORM models (UserORM, DataSourceORM, etc.), use their Pydantic model's from_orm() method instead.
    
    This function reconstructs the appropriate Pydantic model class
    based on the ORM instance's semantic_level and attributes structure.
    
    Args:
        orm_instance: KnowledgeNodeORM instance from database
        pydantic_class: Optional specific Pydantic class to use.
                       If None, will be inferred from semantic_level and attributes.
        
    Returns:
        Validated KnowledgeNode Pydantic model instance
        
    Example:
        >>> orm_node = session.query(KnowledgeNodeORM).first()
        >>> pydantic_node = orm_to_pydantic(orm_node)
    """
    # If no specific class provided, infer from semantic_level
    if pydantic_class is None:
        pydantic_class = _infer_pydantic_class_from_orm(orm_instance)
    
    # Build data dict from ORM
    # Note: children relationship is not loaded to avoid async lazy loading issues
    # Children should be loaded separately using get_children() method if needed
    data = {
        'id': orm_instance.id,
        'source_id': orm_instance.source_id,
        'parent_id': orm_instance.parent_id,
        'semantic_level': orm_instance.semantic_level,
        'physical_level': orm_instance.physical_level,
        'attributes': orm_instance.attributes or {},
        'metadata': orm_instance.node_metadata or {},  # Use node_metadata (ORM attribute name)
        'children': [],  # Children not loaded by default (use get_children() method if needed)
    }
    
    # Create Pydantic instance
    return pydantic_class(**data)


def _infer_pydantic_class_from_orm(orm_instance: KnowledgeNodeORM) -> Type[KnowledgeNodePydantic]:
    """
    Infer Pydantic model class from ORM instance.
    
    Uses semantic_level and attributes structure to determine the correct class.
    When you add new node types, update this function.
    """
    semantic_level = orm_instance.semantic_level
    attributes = orm_instance.attributes or {}
    
    # Convert semantic_level to string if it's an enum
    semantic_level_str = semantic_level.value if hasattr(semantic_level, 'value') else str(semantic_level)
    
    # Structure nodes - check semantic_level
    if semantic_level_str == "CORPUS":
        return CorpusNode
    elif semantic_level_str == "TOPIC":
        return TopicNode
    elif semantic_level_str == "SUBTOPIC":
        return SubtopicNode
    elif semantic_level_str == "DETAIL":
        # For DETAIL nodes, infer from attributes structure
        # Check for specific attribute keys that identify node type
        if 'text' in attributes and 'headers' not in attributes:
            return ParagraphNode
        elif 'url' in attributes or 'file_path' in attributes:
            return ImageNode
        elif 'headers' in attributes or 'rows' in attributes:
            return TableNode
        else:
            # Default to ParagraphNode if unclear
            return ParagraphNode
    
    # Fallback to base class
    return KnowledgeNodePydantic
