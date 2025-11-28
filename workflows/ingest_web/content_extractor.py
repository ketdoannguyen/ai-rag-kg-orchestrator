from __future__ import annotations

import re

from bs4 import BeautifulSoup
from loguru import logger


class ContentExtractor:
    """Extract main content from HTML, removing headers, footers, ads, etc."""

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
        }

        # Patterns to identify navigation/ads in text
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
        """Extract main content markdown from HTML."""
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove unwanted elements
            for tag in self._skip_tags:
                for element in soup.find_all(tag):
                    element.decompose()

            # Remove elements with unwanted classes
            for class_name in self._skip_classes:
                for element in soup.find_all(class_=re.compile(class_name, re.I)):
                    element.decompose()

            # Remove elements with unwanted IDs
            for id_name in self._skip_ids:
                for element in soup.find_all(id=re.compile(id_name, re.I)):
                    element.decompose()

            # Remove elements with data attributes indicating ads
            for element in soup.find_all(attrs={"data-ad": True}):
                element.decompose()
            for element in soup.find_all(attrs={"data-ad-client": True}):
                element.decompose()

            # Try to find main content area - prioritize article/post content
            main_content = (
                soup.find("article")
                or soup.find("main")
                or soup.find("div", class_=re.compile("entry-content|post-content|article-content|content-area|main-content", re.I))
                or soup.find("div", id=re.compile("entry-content|post-content|article-content|content-area|main-content", re.I))
                or soup.find("div", class_=re.compile("content|main|post|article", re.I))
                or soup.find("div", id=re.compile("content|main|post|article", re.I))
            )

            # If we found a content area, try to remove navigation elements from it
            if main_content:
                # Remove navigation elements that might be inside the content area
                for nav in main_content.find_all(["nav", "ul", "ol"], class_=re.compile("menu|nav|breadcrumb", re.I)):
                    nav.decompose()
                # Remove elements with navigation-like classes
                for elem in main_content.find_all(class_=re.compile("menu|nav|breadcrumb|sidebar", re.I)):
                    elem.decompose()

                # Remove links that are likely navigation (short text, common nav words)
                for link in main_content.find_all("a", href=True):
                    link_text = link.get_text(strip=True)
                    href = link.get("href", "")
                    # Remove if it's a navigation link
                    if self._is_navigation_link(link_text, href):
                        link.decompose()

            # Fallback to body if no main content found
            if not main_content:
                main_content = soup.body

            if main_content:
                # Convert to markdown-like text
                markdown = self._html_to_markdown(main_content)
            else:
                # Fallback: use body
                markdown = self._html_to_markdown(soup.body or soup)

            # Clean up markdown
            markdown = self._clean_markdown(markdown)

            # Remove navigation-like content from markdown text
            markdown = self._remove_navigation_text(markdown)

            return markdown
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return ""

    def _is_navigation_link(self, text: str, href: str) -> bool:
        """Check if a link is likely navigation."""
        if not text or len(text) > 50:  # Long text is likely content
            return False

        text_lower = text.lower()
        # Check if text matches navigation patterns
        for pattern in self._nav_patterns:
            if re.search(pattern, text_lower, re.I):
                return True

        # Check if href is navigation-like
        if href.startswith("#") or href.startswith("javascript:"):
            return True

        # Check if it's a common navigation URL pattern
        nav_urls = ["/home", "/about", "/contact", "/login", "/register", "/menu"]
        if any(href.endswith(url) for url in nav_urls):
            return True

        return False

    def _remove_navigation_text(self, markdown: str) -> str:
        """Remove navigation-like content from markdown."""
        lines = markdown.split("\n")
        filtered_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                filtered_lines.append("")
                continue

            # Skip lines that are likely navigation
            if self._is_navigation_line(line):
                continue

            # Skip lines that are just links with common nav text
            if re.match(r"^\*?\s*\[(Trang chủ|Home|Giới thiệu|About|Liên hệ|Contact|Menu|Navigation)\]", line, re.I):
                continue

            filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def _is_navigation_line(self, line: str) -> bool:
        """Check if a line is likely navigation."""
        # Very short lines with navigation keywords
        if len(line) < 30:
            for pattern in self._nav_patterns:
                if re.search(rf"^{pattern}", line, re.I):
                    return True

        # Lines that are just bullet points with links
        if re.match(r"^\s*\*\s*\[.*?\]\(.*?\)\s*$", line):
            # Check if it's a navigation link
            for pattern in self._nav_patterns:
                if re.search(pattern, line, re.I):
                    return True

        return False

    def _html_to_markdown(self, element) -> str:
        """Convert HTML element to markdown text."""
        if element is None:
            return ""

        text_parts = []
        for child in element.children:
            if hasattr(child, "name") and child.name:
                tag_name = child.name.lower()
                if tag_name == "h1":
                    text_parts.append(f"# {child.get_text(strip=True)}\n")
                elif tag_name == "h2":
                    text_parts.append(f"## {child.get_text(strip=True)}\n")
                elif tag_name == "h3":
                    text_parts.append(f"### {child.get_text(strip=True)}\n")
                elif tag_name == "h4":
                    text_parts.append(f"#### {child.get_text(strip=True)}\n")
                elif tag_name == "h5":
                    text_parts.append(f"##### {child.get_text(strip=True)}\n")
                elif tag_name == "h6":
                    text_parts.append(f"###### {child.get_text(strip=True)}\n")
                elif tag_name == "p":
                    text = child.get_text(strip=True)
                    if text:
                        text_parts.append(f"{text}\n")
                elif tag_name == "img":
                    src = child.get("src", "")
                    alt = child.get("alt", "")
                    if src:
                        text_parts.append(f"![{alt}]({src})\n")
                elif tag_name == "a":
                    text = child.get_text(strip=True)
                    href = child.get("href", "")
                    if href:
                        text_parts.append(f"[{text}]({href})")
                    else:
                        text_parts.append(text)
                elif tag_name in ("ul", "ol"):
                    text_parts.append(self._extract_list(child))
                elif tag_name == "li":
                    text = child.get_text(strip=True)
                    if text:
                        text_parts.append(f"- {text}\n")
                elif tag_name == "br":
                    text_parts.append("\n")
                else:
                    # Recursively process other elements
                    text_parts.append(self._html_to_markdown(child))
            else:
                # Text node
                text = str(child).strip()
                if text:
                    text_parts.append(text)

        return "".join(text_parts)

    def _extract_list(self, element) -> str:
        """Extract list items as markdown."""
        items = []
        for li in element.find_all("li", recursive=False):
            text = li.get_text(strip=True)
            if text:
                items.append(f"- {text}\n")
        return "".join(items)

    def _clean_markdown(self, markdown: str) -> str:
        """Clean up markdown text."""
        # Remove excessive newlines
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        # Remove leading/trailing whitespace
        markdown = markdown.strip()
        return markdown
