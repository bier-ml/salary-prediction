import re
from pypdf import PdfReader


class MockNERModel:
    def __init__(self):
        self.entities = {}

    def extract_entities(self, text):
        text = text.lower()  # convert text to lower case
        entities = {}

        patterns = {
            'Пол': r'лет, (.+?)[\n\r]',
            'Желаемая должность': r'\nзарплата (.+?)[\n\r]',
            'График работы': r'график, место работы (.+?)ищу работу в городе:',
            'Место работы': r'ищу работу в городе: (.+?)\.',
            'Стаж работы': r'стаж в желаемой должности (.+?)[\n\r]',
            'Образование': r'основное образование (.+?),',
            'Навыки': r'профессиональные навыки (.+?)(?:основное образование|опыт работы)'
        }

        for entity, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL if entity == 'Навыки' else 0)
            if match:
                entities[entity] = match.group(1).strip().replace('\n', '').replace('.', '').replace(',', '')
            else:
                entities[entity] = ''

        result_str = ' '.join(entities.values()).replace('\n', ' ')

        return entities, result_str

    def predict(self, text):
        extracted_entities = {}

        # Use predefined entities to extract data
        for entity, examples in self.entities.items():
            for example in examples:
                if example.lower() in text:
                    extracted_entities[entity].append(example)

        # Extract additional entities using the extract_entities method
        additional_entities, result_str = self.extract_entities(text)

        # Merge the extracted entities into the result
        for key in additional_entities:
            if key not in extracted_entities:
                extracted_entities[key] = []
            if additional_entities[key]:
                extracted_entities[key].append(additional_entities[key])

        return extracted_entities


class PDFToText:
    def __init__(self, file):
        self.file = file
        self.reader = PdfReader(self.file)

    def extract_text(self):
        text = ""
        for page in self.reader.pages:
            text += page.extract_text()
        return text

    def extract_entities(self, text):
        ner_model = MockNERModel()
        return ner_model.predict(text)
