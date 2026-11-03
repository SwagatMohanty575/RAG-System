"""
table_chunker.py

Builds retrieval chunks from table knowledge objects.

The parser creates row-level structured objects. This chunker keeps rows
complete while adding schema, column-value, and cross-row relationship
text so the embedding model sees both dimensions of the sheet.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from config import TABLE_HEADER_REPEAT, TABLE_ROW_CHUNK_SIZE

from .table_models import TableChunk


class TableChunker:
    def __init__(self) -> None:
        self.row_chunk_size = TABLE_ROW_CHUNK_SIZE
        self.header_repeat = TABLE_HEADER_REPEAT
        self.max_chunk_chars = 6000

    def chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        all_chunks: List[Dict] = []
        global_index = 0

        for page in pages:
            if not page.get("_is_table"):
                all_chunks.append(self._fallback_chunk(page, global_index))
                global_index += 1
                continue

            rows = page.get("_table_rows", [])
            headers = page.get("_headers", [])
            primary_key = page.get("_primary_key")
            workbook = page.get("source_file", "unknown")
            sheet = page.get("section_title", "unknown")
            schema_relationships = page.get("_schema_relationships", [])
            cross_row_relationships = page.get("_cross_row_relationships", [])

            if not rows:
                continue

            for group in self._row_groups(rows):
                group_entities = {row.get("entity") for row in group if row.get("entity")}
                group_cross_relationships = [
                    self._relationship_for_group(rel, group_entities)
                    for rel in cross_row_relationships
                    if group_entities.intersection(set(rel.get("entities", [])))
                ]

                chunk = self._build_chunk(
                    rows=group,
                    headers=headers,
                    primary_key=primary_key,
                    workbook=workbook,
                    sheet=sheet,
                    page_number=page["page_number"],
                    source_file=page["source_file"],
                    chunk_index=global_index,
                    row_start=group[0]["row_index"],
                    row_end=group[-1]["row_index"],
                    schema_relationships=schema_relationships,
                    cross_row_relationships=group_cross_relationships,
                )
                all_chunks.append(chunk.to_dict())
                global_index += 1

        self._link_neighbors(all_chunks)
        return all_chunks

    def _row_groups(self, rows: List[Dict]) -> List[List[Dict]]:
        groups: List[List[Dict]] = []
        current: List[Dict] = []
        current_chars = 0

        for row in rows:
            row_chars = len(str(row.get("semantic_text", ""))) + len(str(row.get("columns", row.get("data", {}))))
            would_exceed_rows = len(current) >= self.row_chunk_size
            would_exceed_chars = current and (current_chars + row_chars) > self.max_chunk_chars

            if would_exceed_rows or would_exceed_chars:
                groups.append(current)
                current = []
                current_chars = 0

            current.append(row)
            current_chars += row_chars

        if current:
            groups.append(current)

        return groups

    def _relationship_for_group(self, relationship: Dict[str, Any], group_entities: set) -> Dict[str, Any]:
        entities = [entity for entity in relationship.get("entities", []) if entity in group_entities]
        filtered = dict(relationship)
        filtered["entities"] = entities

        column = relationship.get("column")
        value = relationship.get("value")
        if column and value and entities:
            filtered["statement"] = f"{column} {value} contains {', '.join(entities)}."

        return filtered

    def _build_chunk(
        self,
        rows: List[Dict],
        headers: List[str],
        primary_key: Optional[str],
        workbook: str,
        sheet: str,
        page_number: int,
        source_file: str,
        chunk_index: int,
        row_start: int,
        row_end: int,
        schema_relationships: List[Dict[str, Any]],
        cross_row_relationships: List[Dict[str, Any]],
    ) -> TableChunk:
        semantic_parts: List[str] = []

        if self.header_repeat and headers:
            semantic_parts.append(f"Workbook: {workbook}. Sheet: {sheet}.")
            semantic_parts.append(f"Columns: {', '.join(headers)}.")
            if primary_key:
                semantic_parts.append(f"Primary entity column: {primary_key}.")

        for rel in schema_relationships:
            if rel.get("type") == "HAS_ATTRIBUTE":
                semantic_parts.append(
                    f"Schema relationship: {rel['source_column']} has attribute {rel['target_column']}."
                )

        row_relationships: List[Dict[str, Any]] = []
        rows_data: List[Dict[str, Any]] = []

        for row in rows:
            rows_data.append(
                {
                    "row_index": row.get("row_index"),
                    "entity": row.get("entity"),
                    "columns": row.get("columns", row.get("data", {})),
                }
            )
            semantic_parts.append(self._row_to_text(row, headers, primary_key))
            row_relationships.extend(row.get("relationships", []))

        for rel in cross_row_relationships:
            statement = rel.get("statement")
            if statement:
                semantic_parts.append(f"Cross-row relationship: {statement}")

        return TableChunk(
            text="\n".join(part for part in semantic_parts if part),
            page_number=page_number,
            section_title=sheet,
            source_file=source_file,
            chunk_index=chunk_index,
            workbook=workbook,
            sheet=sheet,
            row_start=row_start,
            row_end=row_end,
            column_names=headers,
            primary_key=primary_key,
            rows_data=rows_data,
            schema_relationships=schema_relationships,
            row_relationships=row_relationships,
            cross_row_relationships=cross_row_relationships,
            document_id=workbook,
        )

    def _row_to_text(self, row: Dict, headers: List[str], primary_key: Optional[str]) -> str:
        entity = row.get("entity") or f"Row {row.get('row_index')}"
        columns = row.get("columns", row.get("data", {}))
        semantic_text = row.get("semantic_text")

        parts = [
            f"Row {row.get('row_index')} represents entity {entity}.",
            semantic_text or "",
        ]

        for header in headers:
            value = columns.get(header)
            if value is None or str(value).strip() == "":
                continue
            parts.append(f"Column-value relationship: {header} = {value} for {entity}.")
            if header == primary_key:
                parts.append(f"{header} identifies {entity}.")

        return " ".join(part for part in parts if part)

    def _fallback_chunk(self, page: Dict, chunk_index: int) -> Dict:
        return {
            "text": page["text"],
            "page_number": page["page_number"],
            "section_title": page.get("section_title", "Unknown"),
            "source_file": page["source_file"],
            "chunk_index": chunk_index,
        }

    def _link_neighbors(self, chunks: List[Dict]) -> None:
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk["previous_chunk"] = f"chunk_{i - 1}"
            if i < len(chunks) - 1:
                chunk["next_chunk"] = f"chunk_{i + 1}"
