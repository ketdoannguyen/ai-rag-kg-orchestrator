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
from ..models.structure_nodes import StructureNode
from ..models.enums import NodeType
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
    
    # Convert node_type to enum if it's a string (from JSON parsing)
    node_type = pydantic_model.node_type
    if isinstance(node_type, str):
        try:
            node_type = NodeType(node_type)
        except ValueError:
            # If invalid string, try to infer from node type
            if isinstance(pydantic_model, ParagraphNode):
                node_type = NodeType.PARAGRAPH
            elif isinstance(pydantic_model, ImageNode):
                node_type = NodeType.IMAGE
            elif isinstance(pydantic_model, TableNode):
                node_type = NodeType.TABLE
            elif isinstance(pydantic_model, StructureNode):
                node_type = NodeType.STRUCTURE
            else:
                raise ValueError(f"Invalid node_type: {node_type}")
    
    # Create ORM instance
    orm_instance = KnowledgeNodeORM(
        id=data.get("id") or pydantic_model.id,
        source_id=data.get("source_id") or pydantic_model.source_id,
        parent_id=data.get("parent_id") or pydantic_model.parent_id,
        node_type=node_type,
        level_down=pydantic_model.level_down,
        level_up=pydantic_model.level_up,
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
    based on the ORM instance's node_type and attributes structure.
    
    Args:
        orm_instance: KnowledgeNodeORM instance from database
        pydantic_class: Optional specific Pydantic class to use.
                       If None, will be inferred from node_type and attributes.
        
    Returns:
        Validated KnowledgeNode Pydantic model instance
        
    Example:
        >>> orm_node = session.query(KnowledgeNodeORM).first()
        >>> pydantic_node = orm_to_pydantic(orm_node)
    """
    # If no specific class provided, infer from node_type / attributes
    if pydantic_class is None:
        pydantic_class = _infer_pydantic_class_from_orm(orm_instance)
    
    # Build data dict from ORM
    # Note: children relationship is not loaded to avoid async lazy loading issues
    # Children should be loaded separately using get_children() method if needed
    data = {
        "id": orm_instance.id,
        "source_id": orm_instance.source_id,
        "parent_id": orm_instance.parent_id,
        "node_type": orm_instance.node_type,
        "level_down": orm_instance.level_down,
        "level_up": orm_instance.level_up,
        "attributes": orm_instance.attributes or {},
        "metadata": orm_instance.node_metadata or {},  # Use node_metadata (ORM attribute name)
        "children": [],  # Children not loaded by default (use get_children() method if needed)
    }
    
    # Create Pydantic instance
    return pydantic_class(**data)


def _infer_pydantic_class_from_orm(orm_instance: KnowledgeNodeORM) -> Type[KnowledgeNodePydantic]:
    """
    Infer Pydantic model class from ORM instance.
    
    Uses node_type and attributes structure to determine the correct class.
    When you add new node types, update this function.
    """
    attributes = orm_instance.attributes or {}
    
    # Try to infer from node_type first
    node_type = getattr(orm_instance, "node_type", None)
    node_type_str = getattr(node_type, "value", None) or str(node_type) if node_type is not None else ""

    if node_type_str.lower() == "structure":
        # Further refine structure type by level_down if needed
        # but for now just use generic StructureNode
        return StructureNode

    if node_type_str.lower() == "paragraph":
            return ParagraphNode
    if node_type_str.lower() == "image":
            return ImageNode
    if node_type_str.lower() == "table":
            return TableNode

    # Fallback: infer from attributes
    if "text" in attributes and "headers" not in attributes:
            return ParagraphNode
    if "url" in attributes or "file_path" in attributes:
        return ImageNode
    if "headers" in attributes or "rows" in attributes:
        return TableNode
    
    # Ultimate fallback to base class
    return KnowledgeNodePydantic
