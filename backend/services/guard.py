from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from typing import List, Dict
from config import PII_ENTITIES


class PIIGuard:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        self.operators = {
            entity: OperatorConfig("replace", {"new_value": f"[{entity}]"})
            for entity in PII_ENTITIES
        }

    def scrub(self, text: str) -> str:
        if not text or len(text) < 15:
            return text

        results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=PII_ENTITIES,
        )

        if not results:
            return text

        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=self.operators,
        )
        return anonymized.text

    def scrub_chunks(self, chunks: List[Dict]) -> List[Dict]:
        for chunk in chunks:
            chunk["text"] = self.scrub(chunk["text"])
        return chunks

    def scrub_response(self, text: str) -> str:
        return self.scrub(text)