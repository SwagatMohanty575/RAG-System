"""
excel_parser.py

Parses .xlsx and .csv files into table knowledge objects.

Each sheet is treated as a table:
- columns define the schema,
- each row is a complete entity,
- each cell becomes an explicit column -> value relationship,
- repeated values create cross-row relationships,
- semantic text is generated for embeddings while JSON structure is kept.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.parsers.base import BaseParser


class ExcelParser(BaseParser):
    def extract(self, file_path: str) -> List[Dict]:
        ext = Path(file_path).suffix.lower()
        if ext == ".csv":
            return self._extract_csv(file_path)
        return self._extract_excel(file_path)

    def _extract_excel(self, file_path: str) -> List[Dict]:
        import openpyxl

        source = self._source_name(file_path)
        pages: List[Dict] = []
        wb = openpyxl.load_workbook(str(file_path), read_only=False, data_only=True)

        try:
            for sheet_index, sheet_name in enumerate(wb.sheetnames, start=1):
                ws = wb[sheet_name]
                rows = self._worksheet_rows_with_merged_values(ws)

                if rows:
                    pages.append(
                        self._build_table_page(
                            source=source,
                            sheet_name=sheet_name,
                            page_number=sheet_index,
                            rows=rows,
                        )
                    )
        finally:
            wb.close()

        return pages

    def _worksheet_rows_with_merged_values(self, ws) -> List[List[Any]]:
        merged_values: Dict[tuple[int, int], Any] = {}

        for merged_range in ws.merged_cells.ranges:
            top_left_value = ws.cell(
                row=merged_range.min_row,
                column=merged_range.min_col,
            ).value

            for row_idx in range(merged_range.min_row, merged_range.max_row + 1):
                for col_idx in range(merged_range.min_col, merged_range.max_col + 1):
                    merged_values[(row_idx, col_idx)] = top_left_value

        rows: List[List[Any]] = []
        for row_idx in range(1, ws.max_row + 1):
            row_values: List[Any] = []
            for col_idx in range(1, ws.max_column + 1):
                value = ws.cell(row=row_idx, column=col_idx).value
                if self._is_empty(value):
                    value = merged_values.get((row_idx, col_idx), value)
                row_values.append(value)

            if all(self._is_empty(cell) for cell in row_values):
                continue
            rows.append(row_values)

        return rows

    def _extract_csv(self, file_path: str) -> List[Dict]:
        import csv

        source = self._source_name(file_path)
        rows: List[List[str]] = []

        with open(file_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if any(str(cell).strip() for cell in row):
                    rows.append(row)

        if not rows:
            return []

        return [
            self._build_table_page(
                source=source,
                sheet_name="CSV Data",
                page_number=1,
                rows=rows,
            )
        ]

    def _build_table_page(
        self,
        source: str,
        sheet_name: str,
        page_number: int,
        rows: List[List[Any]],
    ) -> Dict:
        headers = self._normalize_headers(rows[0])
        data_rows = rows[1:]
        primary_key = self._detect_primary_key(headers)
        schema_relationships = self._build_schema_relationships(headers, primary_key)

        structured_rows: List[Dict] = []
        value_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        for offset, row in enumerate(data_rows, start=2):
            row_data = self._row_to_dict(headers, row)
            if all(self._is_empty(value) for value in row_data.values()):
                continue

            entity = self._entity_name(row_data, primary_key, offset)
            relationships = self._build_row_relationships(entity, row_data, primary_key)
            semantic_text = self._generate_semantic_text(
                entity=entity,
                row_data=row_data,
                primary_key=primary_key,
                relationships=relationships,
            )

            for column, value in row_data.items():
                if self._is_empty(value):
                    continue
                value_index[column][self._value_to_text(value)].append(entity)

            structured_rows.append(
                {
                    "row_index": offset,
                    "entity": entity,
                    "columns": row_data,
                    "data": row_data,  # Backward compatible with TableChunker callers.
                    "semantic_text": semantic_text,
                    "relationships": relationships,
                }
            )

        cross_row_relationships = self._build_cross_row_relationships(
            value_index=value_index,
            primary_key=primary_key,
        )

        sheet_text = self._build_sheet_semantic_text(
            sheet_name=sheet_name,
            headers=headers,
            primary_key=primary_key,
            schema_relationships=schema_relationships,
            rows=structured_rows,
            cross_row_relationships=cross_row_relationships,
        )

        return {
            "page_number": page_number,
            "text": sheet_text,
            "section_title": sheet_name,
            "source_file": source,
            "_is_table": True,
            "_headers": headers,
            "_primary_key": primary_key,
            "_table_rows": structured_rows,
            "_schema_relationships": schema_relationships,
            "_cross_row_relationships": cross_row_relationships,
            "_total_rows": len(structured_rows),
        }

    def _normalize_headers(self, raw_headers: List[Any]) -> List[str]:
        headers: List[str] = []
        seen: Dict[str, int] = {}

        for index, header in enumerate(raw_headers, start=1):
            name = str(header).strip() if header is not None else ""
            if not name:
                name = f"Column {index}"

            count = seen.get(name, 0) + 1
            seen[name] = count
            headers.append(name if count == 1 else f"{name} {count}")

        return headers

    def _row_to_dict(self, headers: List[str], row: List[Any]) -> Dict[str, Any]:
        row_data: Dict[str, Any] = {}
        for col_idx, header in enumerate(headers):
            value = row[col_idx] if col_idx < len(row) else None
            row_data[header] = self._clean_value(value)
        return row_data

    def _detect_primary_key(self, headers: List[str]) -> Optional[str]:
        keywords = ("id", "name", "employee", "project", "department", "code", "key", "title")
        for header in headers:
            normalized = header.lower()
            if any(keyword in normalized for keyword in keywords):
                return header
        return headers[0] if headers else None

    def _entity_name(self, row_data: Dict[str, Any], primary_key: Optional[str], row_index: int) -> str:
        if primary_key and not self._is_empty(row_data.get(primary_key)):
            return self._value_to_text(row_data[primary_key])
        return f"Row {row_index}"

    def _build_schema_relationships(
        self,
        headers: List[str],
        primary_key: Optional[str],
    ) -> List[Dict[str, str]]:
        relationships: List[Dict[str, str]] = []
        if not primary_key:
            return relationships

        for header in headers:
            if header == primary_key:
                continue
            relationships.append(
                {
                    "type": "HAS_ATTRIBUTE",
                    "source_column": primary_key,
                    "target_column": header,
                }
            )

        for left, right in zip(headers, headers[1:]):
            relationships.append(
                {
                    "type": "NEXT_COLUMN",
                    "source_column": left,
                    "target_column": right,
                }
            )

        return relationships

    def _build_row_relationships(
        self,
        entity: str,
        row_data: Dict[str, Any],
        primary_key: Optional[str],
    ) -> List[Dict[str, Any]]:
        relationships: List[Dict[str, Any]] = []

        for column, value in row_data.items():
            if self._is_empty(value):
                continue

            relationship_type = "IDENTIFIED_BY" if column == primary_key else self._relationship_type(column)
            relationships.append(
                {
                    "type": relationship_type,
                    "source": entity,
                    "column": column,
                    "target": value,
                    "statement": f"{column} = {self._value_to_text(value)}",
                }
            )

        return relationships

    def _build_cross_row_relationships(
        self,
        value_index: Dict[str, Dict[str, List[str]]],
        primary_key: Optional[str],
    ) -> List[Dict[str, Any]]:
        relationships: List[Dict[str, Any]] = []

        for column, values in value_index.items():
            if column == primary_key:
                continue
            for value, entities in values.items():
                if len(entities) < 2:
                    continue
                relationships.append(
                    {
                        "type": "SHARED_VALUE",
                        "column": column,
                        "value": value,
                        "entities": entities,
                        "statement": f"{column} {value} contains {', '.join(entities)}.",
                    }
                )

        return relationships

    def _generate_semantic_text(
        self,
        entity: str,
        row_data: Dict[str, Any],
        primary_key: Optional[str],
        relationships: List[Dict[str, Any]],
    ) -> str:
        sentences = [f"Entity {entity} is a row in the table."]

        for rel in relationships:
            column = rel["column"]
            value = self._value_to_text(rel["target"])
            if column == primary_key:
                sentences.append(f"{column} identifies {entity}.")
            else:
                sentences.append(f"{entity}'s {column} is {value}.")

        return " ".join(sentences)

    def _build_sheet_semantic_text(
        self,
        sheet_name: str,
        headers: List[str],
        primary_key: Optional[str],
        schema_relationships: List[Dict[str, str]],
        rows: List[Dict],
        cross_row_relationships: List[Dict],
    ) -> str:
        parts = [
            f"Sheet {sheet_name} has columns: {', '.join(headers)}.",
            f"The primary entity column is {primary_key}." if primary_key else "No primary entity column was detected.",
        ]

        for rel in schema_relationships:
            if rel["type"] == "HAS_ATTRIBUTE":
                parts.append(f"{rel['source_column']} has attribute {rel['target_column']}.")

        parts.extend(row["semantic_text"] for row in rows)
        parts.extend(rel["statement"] for rel in cross_row_relationships)
        return "\n".join(parts)

    def _relationship_type(self, column: str) -> str:
        normalized = column.lower()
        if "department" in normalized:
            return "WORKS_IN"
        if "salary" in normalized or "pay" in normalized or "compensation" in normalized:
            return "EARNS"
        if "experience" in normalized:
            return "HAS_EXPERIENCE"
        if "project" in normalized:
            return "WORKS_ON"
        if "manager" in normalized or "lead" in normalized:
            return "REPORTS_TO"
        return "HAS_ATTRIBUTE"

    def _clean_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    def _value_to_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _is_empty(self, value: Any) -> bool:
        return value is None or str(value).strip() == ""
