"""Split paragraphs into sentences - each sentence becomes a separate paragraph."""

from __future__ import annotations

import re


class SemanticParagraphSplitter:
    """Split paragraphs into sentences - each sentence becomes a separate paragraph."""

    def split_paragraph(self, text: str) -> list[str]:
        """
        Split a paragraph into sentences. Each sentence becomes a separate paragraph.
        Splits by sentence-ending punctuation (. ! ?) and newlines (\n).

        Args:
            text: The paragraph text to split

        Returns:
            List of sentences (each sentence is a paragraph)
        """
        if not text or not text.strip():
            return []

        # First, split by newlines to handle multi-line paragraphs
        lines = text.split("\n")

        # Process each line and split by sentence boundaries
        all_sentences = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Split by sentence boundaries (., !, ?)
            sentences = self._split_by_sentences(line)
            all_sentences.extend(sentences)

        # Filter out empty sentences
        all_sentences = [s.strip() for s in all_sentences if s.strip()]

        # If no sentences found, return original text as single paragraph
        if not all_sentences:
            return [text.strip()] if text.strip() else []

        return all_sentences

    def _split_by_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences based on sentence-ending punctuation (. ! ?).

        Handles Vietnamese and English sentences.
        Each sentence becomes a separate paragraph.
        """
        # Unified pattern: Match . ! ? followed by:
        # 1. Space + capital letter (new sentence)
        # 2. Newline (end of line)
        # 3. End of text ($)
        # This handles all cases: normal sentences, list items, end of text
        sentence_pattern = r"([.!?]+)(?=\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]|\s*\n|$)"

        # Split text while keeping the punctuation
        parts = re.split(sentence_pattern, text)

        if len(parts) < 2:
            # No sentence boundaries found, return as single sentence
            return [text] if text.strip() else []

        # Reconstruct sentences with their punctuation
        sentences = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts):
                # Combine text + punctuation
                sentence = parts[i] + parts[i + 1]
                if sentence.strip():
                    sentences.append(sentence.strip())
                i += 2
            else:
                # Last part without punctuation
                if parts[i].strip():
                    sentences.append(parts[i].strip())
                i += 1

        # Filter out very short sentences (likely artifacts)
        sentences = [s for s in sentences if len(s.strip()) > 3]

        return sentences if sentences else [text] if text.strip() else []
