import re
from collections import defaultdict

from pypdf import PdfReader


class MockNERModel:
    def __init__(self):
        self.entities = {}

    @staticmethod
    def extract_entities(text: str) -> tuple[dict[str, str], str]:
        text = text.lower()
        entities = {}

        patterns = {
            "Пол": r"лет, (.+?)[\n\r]",
            "Желаемая должность": r"\nзарплата (.+?)[\n\r]",
            "График работы": r"график, место работы (.+?)ищу работу в городе:",
            "Место работы": r"ищу работу в городе: (.+?)\.",
            "Стаж работы": r"стаж в желаемой должности (.+?)[\n\r]",
            "Образование": r"основное образование (.+?),",
            "Навыки": r"профессиональные навыки (.+?)(?:основное образование|опыт работы)",
        }

        for entity, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL if entity == "Навыки" else 0)
            if match:
                entities[entity] = match.group(1).strip().replace("\n", "").replace(".", "").replace(",", "")
            else:
                entities[entity] = ""

        result_str = " ".join(entities.values()).replace("\n", " ")

        return entities, result_str

    def predict(self, text: str) -> defaultdict[str, list]:
        extracted_entities = defaultdict(list)

        for entity, examples in self.entities.items():
            for example in filter(lambda ex: ex.lower() in text, examples):
                extracted_entities[entity].append(example)

        additional_entities, _ = self.extract_entities(text)
        for key, value in additional_entities.items():
            extracted_entities[key].append(value)

        return extracted_entities


class PDFToText:
    def __init__(self, file):
        self.file = file
        self.reader = PdfReader(self.file)

    def extract_text(self) -> str:
        text = ""
        for page in self.reader.pages:
            text += page.extract_text()
        return text

    @staticmethod
    def extract_entities(text: str) -> defaultdict[str, list]:
        ner_model = MockNERModel()
        return ner_model.predict(text)
