"""Extract main content from HTML with multi-layer fallback strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from bs4 import BeautifulSoup, Tag
from loguru import logger


class ExtractionStrategy(str, Enum):
    """Content extraction strategy used."""

    SEMANTIC = "semantic"  # <article>, <main>, semantic containers
    READABILITY = "readability"  # Text density scoring
    BODY = "body"  # Full body extraction
    FAILED = "failed"  # Extraction failed


@dataclass
class ExtractionResult:
    """Result from content extraction."""

    content: str
    strategy: ExtractionStrategy
    score: float = 0.0  # Content quality score (0-1)
    stats: dict[str, Any] | None = None


class ContentExtractor:
    """Extract main content from HTML with multiple fallback strategies."""

    # Minimum content length to consider extraction successful
    MIN_CONTENT_LENGTH = 100

    # Minimum text density ratio (text chars / total chars including tags)
    MIN_TEXT_DENSITY = 0.3

    def __init__(self) -> None:
        self._skip_tags = {
            "header",
            "footer",
            "nav",
            "aside",
            "script",
            "style",
            "noscript",
            "iframe",
            "embed",
            "object",
            "form",
        }
        self._skip_classes = {
            "header",
            "footer",
            "nav",
            "sidebar",
            "ad",
            "advertisement",
            "ads",
            "advert",
            "banner",
            "popup",
            "modal",
            "cookie",
            "newsletter",
            "social",
            "share",
            "comment",
            "comments",
            "widget",
            "related",
            "recommended",
            "trending",
            "popular",
            "menu",
            "navigation",
            "breadcrumb",
        }
        self._skip_ids = {
            "header",
            "footer",
            "nav",
            "sidebar",
            "ad",
            "advertisement",
            "ads",
            "popup",
            "modal",
            "cookie",
            "menu",
            "navigation",
        }
        self._nav_patterns = [
            r"Trang chủ",
            r"Giới thiệu",
            r"Liên hệ",
            r"Đăng nhập",
            r"Đăng ký",
            r"Menu",
            r"Navigation",
            r"Home",
            r"About",
            r"Contact",
        ]

    def extract_main_content(self, html: str, url: str) -> str:
        """Extract main content markdown from HTML using multi-layer strategies.

        Tries strategies in order: semantic -> readability -> body fallback
        Returns best result based on content quality.
        """
        result = self.extract_with_strategy(html, url)

        logger.debug(
            f"Content extraction completed",
            strategy=result.strategy.value,
            content_length=len(result.content),
            score=f"{result.score:.2f}",
            url=url[:50] + "..." if len(url) > 50 else url,
        )

        return result.content

    def extract_with_strategy(self, html: str, url: str) -> ExtractionResult:
        """Extract content using multiple strategies and return best result."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            self._clean_unwanted_elements(soup)

            # Try strategies in priority order
            strategies = [
                (ExtractionStrategy.SEMANTIC, self._extract_semantic),
                (ExtractionStrategy.READABILITY, self._extract_readability),
                (ExtractionStrategy.BODY, self._extract_body),
            ]

            best_result: ExtractionResult | None = None

            for strategy_type, strategy_func in strategies:
                try:
                    content = strategy_func(soup)
                    if content and len(content.strip()) >= self.MIN_CONTENT_LENGTH:
                        score = self._score_content(content)
                        result = ExtractionResult(
                            content=content,
                            strategy=strategy_type,
                            score=score,
                            stats={
                                "length": len(content),
                                "lines": len(content.splitlines()),
                                "headings": len(re.findall(r"^#+\s", content, re.MULTILINE)),
                            },
                        )

                        # Use this result if it's better
                        if best_result is None or result.score > best_result.score:
                            best_result = result

                        # If score is high enough, stop trying other strategies
                        if result.score >= 0.7:
                            break

                except Exception as e:
                    logger.warning(f"Strategy {strategy_type.value} failed: {e}")
                    continue

            if best_result:
                return best_result

            return ExtractionResult(
                content="",
                strategy=ExtractionStrategy.FAILED,
                score=0.0,
            )

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return ExtractionResult(
                content="",
                strategy=ExtractionStrategy.FAILED,
                score=0.0,
            )

    def _clean_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements from soup in place."""
        # Remove unwanted tags
        for tag in self._skip_tags:
            for element in soup.find_all(tag):
                element.decompose()

        # Remove elements with unwanted classes
        for class_name in self._skip_classes:
            for element in soup.find_all(class_=re.compile(rf"\b{class_name}\b", re.I)):
                element.decompose()

        # Remove elements with unwanted IDs
        for id_name in self._skip_ids:
            for element in soup.find_all(id=re.compile(rf"\b{id_name}\b", re.I)):
                element.decompose()

        # Remove ad elements
        for element in soup.find_all(attrs={"data-ad": True}):
            element.decompose()
        for element in soup.find_all(attrs={"data-ad-client": True}):
            element.decompose()

    def _extract_semantic(self, soup: BeautifulSoup) -> str:
        """Strategy 1: Extract from semantic HTML containers."""
        # Priority order for semantic containers
        selectors = [
            ("article", {}),
            ("main", {}),
            ("[role=main]", {}),
            ("[role=article]", {}),
            (
                "div",
                {
                    "class_": re.compile(
                        r"entry-content|post-content|article-content|article-body", re.I
                    )
                },
            ),
            (
                "div",
                {
                    "id": re.compile(
                        r"entry-content|post-content|article-content|article-body", re.I
                    )
                },
            ),
            ("div", {"class_": re.compile(r"content-area|main-content|single-content", re.I)}),
            ("div", {"class_": re.compile(r"\bcontent\b|\bmain\b|\bpost\b|\barticle\b", re.I)}),
        ]

        for tag_name, attrs in selectors:
            if tag_name.startswith("["):
                # CSS selector-like syntax
                attr_match = re.match(r"\[(\w+)=(\w+)\]", tag_name)
                if attr_match:
                    element = soup.find(attrs={attr_match.group(1): attr_match.group(2)})
                else:
                    continue
            else:
                element = soup.find(tag_name, **attrs)

            if element and isinstance(element, Tag):
                self._clean_navigation_in_content(element)
                markdown = self._html_to_markdown(element)
                markdown = self._clean_markdown(markdown)
                if len(markdown.strip()) >= self.MIN_CONTENT_LENGTH:
                    return markdown

        return ""

    def _extract_readability(self, soup: BeautifulSoup) -> str:
        """Strategy 2: Find content area by text density scoring."""
        candidates = []

        # Find all container elements
        for container in soup.find_all(["div", "section", "article", "main"]):
            if not isinstance(container, Tag):
                continue

            # Calculate text density
            text = container.get_text(strip=True)
            html_length = len(str(container))

            if html_length == 0:
                continue

            text_density = len(text) / html_length

            # Score based on multiple factors
            score = self._calculate_container_score(container, text, text_density)

            if score > 0 and len(text) >= self.MIN_CONTENT_LENGTH:
                candidates.append((container, score, len(text)))

        if not candidates:
            return ""

        # Sort by score, then by text length
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # Use best candidate
        best_container = candidates[0][0]
        self._clean_navigation_in_content(best_container)
        markdown = self._html_to_markdown(best_container)
        return self._clean_markdown(markdown)

    def _extract_body(self, soup: BeautifulSoup) -> str:
        """Strategy 3: Extract from body as fallback."""
        body = soup.body or soup
        if body:
            self._clean_navigation_in_content(body)
            markdown = self._html_to_markdown(body)
            return self._clean_markdown(markdown)
        return ""

    def _calculate_container_score(self, container: Tag, text: str, text_density: float) -> float:
        """Calculate score for a potential content container."""
        score = 0.0

        # Base score from text density
        if text_density >= 0.5:
            score += 3.0
        elif text_density >= 0.3:
            score += 2.0
        elif text_density >= 0.2:
            score += 1.0

        # Bonus for semantic elements
        if container.name in ("article", "main"):
            score += 2.0

        # Bonus for content-related class names
        class_names = " ".join(container.get("class", []))
        if re.search(r"content|article|post|entry|body|text", class_names, re.I):
            score += 1.5

        # Bonus for having paragraphs
        p_count = len(container.find_all("p"))
        if p_count >= 3:
            score += 1.0

        # Bonus for having headings
        h_count = len(container.find_all(re.compile(r"^h[1-6]$")))
        if h_count >= 1:
            score += 1.0

        # Penalty for too many links (navigation-like)
        link_count = len(container.find_all("a"))
        text_words = len(text.split())
        if text_words > 0:
            link_ratio = link_count / text_words
            if link_ratio > 0.3:  # More than 30% links
                score -= 2.0

        # Penalty for navigation-like classes
        if re.search(r"nav|menu|sidebar|widget", class_names, re.I):
            score -= 3.0

        return score

    def _clean_navigation_in_content(self, element: Tag) -> None:
        """Remove navigation elements inside content area."""
        # Remove navigation elements
        for nav in element.find_all(
            ["nav", "ul", "ol"], class_=re.compile(r"menu|nav|breadcrumb", re.I)
        ):
            nav.decompose()

        # Remove elements with navigation-like classes
        for elem in element.find_all(class_=re.compile(r"menu|nav|breadcrumb|sidebar", re.I)):
            elem.decompose()

        # Remove navigation links
        for link in element.find_all("a", href=True):
            link_text = link.get_text(strip=True)
            href = link.get("href", "")
            if self._is_navigation_link(link_text, href):
                link.decompose()

    def _is_navigation_link(self, text: str, href: str) -> bool:
        """Check if a link is likely navigation."""
        if not text or len(text) > 50:
            return False

        text_lower = text.lower()
        for pattern in self._nav_patterns:
            if re.search(pattern, text_lower, re.I):
                return True

        if href.startswith("#") or href.startswith("javascript:"):
            return True

        nav_urls = ["/home", "/about", "/contact", "/login", "/register", "/menu"]
        if any(href.endswith(url) for url in nav_urls):
            return True

        return False

    def _score_content(self, content: str) -> float:
        """Score content quality from 0.0 to 1.0."""
        if not content:
            return 0.0

        score = 0.0

        # Length score (max 0.3)
        length = len(content)
        if length >= 5000:
            score += 0.3
        elif length >= 2000:
            score += 0.2
        elif length >= 500:
            score += 0.1

        # Heading score (max 0.3)
        headings = len(re.findall(r"^#+\s", content, re.MULTILINE))
        if headings >= 5:
            score += 0.3
        elif headings >= 2:
            score += 0.2
        elif headings >= 1:
            score += 0.1

        # Paragraph structure score (max 0.2)
        paragraphs = len([p for p in content.split("\n\n") if len(p.strip()) > 50])
        if paragraphs >= 5:
            score += 0.2
        elif paragraphs >= 3:
            score += 0.1

        # Link ratio penalty (max -0.2)
        links = len(re.findall(r"\[.*?\]\(.*?\)", content))
        words = len(content.split())
        if words > 0:
            link_ratio = links / words
            if link_ratio > 0.2:
                score -= 0.2
            elif link_ratio > 0.1:
                score -= 0.1

        # Clean text bonus (max 0.2)
        # Text without markdown formatting should be readable
        clean_text = re.sub(r"[#*\[\]()]", "", content)
        clean_ratio = len(clean_text.strip()) / max(len(content), 1)
        if clean_ratio >= 0.8:
            score += 0.2
        elif clean_ratio >= 0.6:
            score += 0.1

        return min(max(score, 0.0), 1.0)

    def _html_to_markdown(self, element: Tag) -> str:
        """Convert HTML element to markdown text."""
        if element is None:
            return ""

        text_parts = []
        for child in element.children:
            if hasattr(child, "name") and child.name:
                tag_name = child.name.lower()
                if tag_name == "h1":
                    text_parts.append(f"# {child.get_text(strip=True)}\n\n")
                elif tag_name == "h2":
                    text_parts.append(f"## {child.get_text(strip=True)}\n\n")
                elif tag_name == "h3":
                    text_parts.append(f"### {child.get_text(strip=True)}\n\n")
                elif tag_name == "h4":
                    text_parts.append(f"#### {child.get_text(strip=True)}\n\n")
                elif tag_name == "h5":
                    text_parts.append(f"##### {child.get_text(strip=True)}\n\n")
                elif tag_name == "h6":
                    text_parts.append(f"###### {child.get_text(strip=True)}\n\n")
                elif tag_name == "p":
                    text = child.get_text(strip=True)
                    if text:
                        text_parts.append(f"{text}\n\n")
                elif tag_name == "img":
                    src = child.get("src", "")
                    alt = child.get("alt", "")
                    if src:
                        text_parts.append(f"![{alt}]({src})\n\n")
                elif tag_name == "a":
                    text = child.get_text(strip=True)
                    href = child.get("href", "")
                    if href and text:
                        text_parts.append(f"[{text}]({href})")
                    elif text:
                        text_parts.append(text)
                elif tag_name in ("ul", "ol"):
                    text_parts.append(self._extract_list(child))
                elif tag_name == "li":
                    text = child.get_text(strip=True)
                    if text:
                        text_parts.append(f"- {text}\n")
                elif tag_name == "br":
                    text_parts.append("\n")
                elif tag_name == "blockquote":
                    quote_text = child.get_text(strip=True)
                    if quote_text:
                        text_parts.append(f"> {quote_text}\n\n")
                elif tag_name == "pre":
                    code_text = child.get_text()
                    if code_text:
                        text_parts.append(f"```\n{code_text}\n```\n\n")
                elif tag_name == "code":
                    code_text = child.get_text(strip=True)
                    if code_text:
                        text_parts.append(f"`{code_text}`")
                elif tag_name in ("strong", "b"):
                    text = child.get_text(strip=True)
                    if text:
                        text_parts.append(f"**{text}**")
                elif tag_name in ("em", "i"):
                    text = child.get_text(strip=True)
                    if text:
                        text_parts.append(f"*{text}*")
                elif tag_name in ("div", "section", "span"):
                    # Recursively process container elements
                    text_parts.append(self._html_to_markdown(child))
                else:
                    # Other elements - just get text
                    text = child.get_text(strip=True)
                    if text:
                        text_parts.append(f"{text}\n")
            else:
                # Text node
                text = str(child).strip()
                if text:
                    text_parts.append(text)

        return "".join(text_parts)

    def _extract_list(self, element: Tag) -> str:
        """Extract list items as markdown."""
        items = []
        for li in element.find_all("li", recursive=False):
            text = li.get_text(strip=True)
            if text:
                items.append(f"- {text}\n")
        return "".join(items) + "\n" if items else ""

    def _clean_markdown(self, markdown: str) -> str:
        """Clean up markdown text."""
        # Remove excessive newlines
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in markdown.splitlines()]
        markdown = "\n".join(lines)
        # Remove leading/trailing whitespace
        markdown = markdown.strip()
        return markdown
