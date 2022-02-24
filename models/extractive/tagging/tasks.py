import logging
import os

from typing import List, TextIO, Union
from models.extractive.tagging.utils_token_classification import InputExample, Split, TokenClassificationTask

logger = logging.getLogger(__name__)


class ETAGGING(TokenClassificationTask):
    def read_examples_from_file(self, data_dir, mode: Union[Split, str]) -> List[InputExample]:
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")

        guid_index = 1
        examples = []
        with open(file_path, encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                if line == "\n" and len(words) > 0:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
                else:
                    d = line.rstrip("\n").split()
                    if len(d) == 2:
                        words.append(d[0])
                        labels.append(d[1])

        return examples

    def write_predictions_to_file(self, writer: TextIO, test_input_reader: TextIO, preds_list: List):
        example_id = 0
        for line in test_input_reader:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                writer.write(line)
                if example_id < len(preds_list) and not preds_list[example_id]:
                    example_id += 1
            elif example_id < len(preds_list) and preds_list[example_id]:
                output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                writer.write(output_line)

    def get_labels(self, path: str) -> List[str]:
        if path:
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        else:
            return ["O", "B-FACET", "I-FACET"]
