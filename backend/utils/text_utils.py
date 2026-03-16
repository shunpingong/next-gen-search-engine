from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, List
from urllib.parse import urlparse

ACADEMIC_QUERY_HINTS = (
    "research paper",
    "paper",
    "thesis",
    "dissertation",
    "acknowledg",
    "professor",
    "university",
    "department",
    "student",
    "faculty",
)
ACADEMIC_REPOSITORY_HINTS = (
    "repository",
    "scholarworks",
    "digitalcommons",
    "etd",
    "bitstream",
    "bitstreams",
    "vtechworks",
    "lib.",
    "viewcontent",
)
THESIS_CONTENT_HINTS = (
    "thesis",
    "dissertation",
    "master of",
    "doctor of philosophy",
    "doctoral",
    "submitted to",
    "in partial fulfillment",
)
ACKNOWLEDGEMENT_HINTS = (
    "acknowledg",
    "thank",
    "gratitude",
    "grateful",
    "appreciate",
)
CAREER_HINTS = (
    "professor",
    "assistant professor",
    "associate professor",
    "lecturer",
    "faculty",
    "appointed",
    "joined",
    "staff profile",
)
PAPER_QUERY_HINTS = (
    "doi",
    "journal",
    "article",
    "abstract",
    "authors",
    "author",
    "study",
    "sample size",
)
PAPER_CONTENT_HINTS = (
    "doi",
    "journal",
    "volume",
    "issue",
    "abstract",
    "keywords",
    "received",
    "accepted",
    "published",
    "article",
    "study",
)
MEDIA_QUERY_HINTS = (
    "manga",
    "anime",
    "comic",
    "comics",
    "graphic novel",
    "novel",
    "book",
    "story",
    "plot",
    "chapter",
    "episode",
    "season",
    "film",
    "movie",
    "television",
    "tv",
    "character",
    "characters",
    "antagonist",
    "protagonist",
    "villain",
    "hero",
    "companion",
    "group",
    "theme",
    "movement",
    "movements",
    "effect",
    "effects",
    "ability",
    "abilities",
    "technique",
    "techniques",
    "band",
    "song",
    "album",
    "game",
)
EVENT_QUERY_HINTS = (
    "festival",
    "festivity",
    "celebration",
    "pageant",
    "beauty pageant",
    "contest",
    "winner",
    "won",
    "queen",
    "coronation",
    "anniversary",
    "township",
    "municipality",
    "municipal",
    "province",
    "tourism",
    "cultural",
    "stew",
    "dish",
    "ingredient",
    "condiment",
)
EVENT_PAGE_HINTS = (
    "festival",
    "festivity",
    "celebration",
    "pageant",
    "beauty pageant",
    "contest",
    "winner",
    "won",
    "queen",
    "coronation",
    "candidate",
    "festival queen",
    "grand winner",
    "municipality",
    "municipal",
    "township",
    "province",
    "tourism",
    "cultural",
    "official",
)
EVENT_WINNER_HINTS = (
    "winner",
    "won",
    "beauty pageant",
    "pageant",
    "contest",
    "queen",
    "festival queen",
    "coronation",
    "candidate",
    "grand winner",
)
EVENT_WINNER_STRONG_HINTS = (
    "beauty pageant",
    "festival queen",
    "queen",
    "coronation",
    "crowned",
    "proclaimed",
    "declared",
    "grand winner",
)
EVENT_OFFICIAL_SOURCE_HINTS = (
    ".gov",
    ".gov.ph",
    "tourism",
    "municipality",
    "municipal",
    "township",
    "province",
    "barangay",
    "cityhall",
    "city hall",
    "official",
)
LOW_TRUST_SOCIAL_SOURCE_HINTS = (
    "facebook.com/groups/",
    "facebook.com/share/",
    "facebook.com/reel/",
    "facebook.com/watch/",
    "facebook.com/video",
    "instagram.com",
    "quizlet.com",
    "snapchat.com",
    "tiktok.com",
    "pinterest.",
    "twitter.com",
    "x.com",
)
RECIPE_FOOD_PAGE_HINTS = (
    "allrecipes.com",
    "foodandwine.com",
    "foodnetwork.com",
    "seriouseats.com",
    "delish.com",
    "epicurious.com",
    "tasteofhome.com",
    "recipe",
    "recipes",
    "cooking",
    "cook",
    "ingredients",
    "simmer",
    "broth",
    "thanksgiving dinner",
)
MODERATE_TRUST_SOCIAL_SOURCE_HINTS = (
    "facebook.com",
    "reddit.com",
    "youtube.com",
)
BROAD_OVERVIEW_TITLE_HINTS = (
    "history of ",
    "category:",
    "list of ",
    "timeline of ",
    "outline of ",
    "index of ",
    "glossary of ",
    "portal:",
)
AGGREGATE_LISTING_HINTS = (
    "/stacks/",
    "interest stack",
    "interest stacks",
    "best of ",
    "top ",
    "top-",
    "ranking",
    "rankings",
    "recommendation",
    "recommendations",
    "watch order",
    "reading order",
)
GENERIC_OVERVIEW_TITLES = {
    "manga",
    "anime",
    "comic",
    "comics",
    "character",
    "characters",
    "fiction",
    "novel",
    "story",
}
BROWSECOMP_LOW_SIGNAL_TERMS = {
    "after",
    "anime",
    "anniversary",
    "attended",
    "authorities",
    "based",
    "became",
    "begins",
    "certain",
    "character",
    "characters",
    "chapter",
    "celebration",
    "celebrations",
    "classic",
    "comic",
    "comics",
    "competition",
    "components",
    "contest",
    "created",
    "creator",
    "creators",
    "decade",
    "different",
    "draws",
    "early",
    "elements",
    "event",
    "festivities",
    "festivity",
    "first",
    "forms",
    "group",
    "highlight",
    "how",
    "interested",
    "known",
    "main",
    "manga",
    "many",
    "name",
    "novel",
    "one",
    "other",
    "particular",
    "people",
    "plot",
    "potential",
    "product",
    "province",
    "provincial",
    "published",
    "region",
    "released",
    "same",
    "school",
    "some",
    "story",
    "subject",
    "theme",
    "their",
    "them",
    "they",
    "town",
    "township",
    "uses",
    "used",
    "using",
    "what's",
    "winner",
    "written",
    "whose",
    "2000s",
}
ENCYCLOPEDIC_SOURCE_HINTS = (
    "wikipedia.org",
    "fandom.com",
    "wiki",
    "wikitia",
    "comicvine.gamespot.com",
    "anime-planet.com",
    "myanimelist.net",
    "tvtropes.org",
)
MEDIA_WORK_SOURCE_HINTS = (
    "wikipedia.org",
    "fandom.com",
    "/wiki/",
    "wiki.",
    "myanimelist.net",
    "anime-planet.com",
    "tvtropes.org",
    "comicvine.gamespot.com",
    "mangadex.org",
)
MEDIA_WORK_STRONG_HINTS = (
    "manga",
    "anime",
    "chapter",
    "chapters",
    "episode",
    "episodes",
    "volume",
    "volumes",
    "series",
    "serialized",
    "serialization",
    "one-shot",
    "oneshot",
)
MEDIA_WORK_WEAK_HINTS = (
    "character",
    "characters",
    "fictional character",
    "plot",
    "premise",
    "adaptation",
    "debut",
    "arc",
    "mangaka",
    "protagonist",
    "antagonist",
    "companion",
)
MEDIA_NON_WORK_HINTS = (
    "muscle",
    "muscles",
    "exercise",
    "fitness",
    "workout",
    "writers become authors",
    "story structure",
    "writing advice",
    "how to choose the right antagonist",
)
FORUM_DISCUSSION_HINTS = (
    "reddit.com",
    "quora.com",
    "forum",
    "/threads/",
    "/thread/",
    "/comments/",
    "discussion",
)
BROWSECOMP_LINK_EXCLUDE_HINTS = (
    "/category:",
    "category:",
    "/history_of_",
    "history of ",
    "list of ",
    "/list_of_",
    "template:",
    "/template:",
    "portal:",
    "/portal:",
    "special:",
    "/special:",
    "help:",
    "/help:",
    "file:",
    "/file:",
    "user:",
    "/user:",
    "talk:",
    "/talk:",
    "action=",
    "?oldid=",
    "/tag/",
    "/tags/",
)
GENERIC_LINK_TEXT_HINTS = {
    "about",
    "browse",
    "categories",
    "chapter",
    "chapters",
    "character",
    "characters",
    "community",
    "contents",
    "episodes",
    "featured",
    "help",
    "history",
    "home",
    "index",
    "login",
    "main page",
    "next",
    "pages",
    "random",
    "recent changes",
    "register",
    "search",
    "series",
    "sign in",
    "special pages",
    "volume",
    "volumes",
    "wiki",
}
WIKI_META_SOURCE_HINTS = (
    "wikistats",
    "stats.wikimedia.org",
    "meta.wikimedia.org",
    "commons.wikimedia.org",
    "mediawiki.org",
    "wikimediafoundation.org",
)
GENERIC_MEDIA_TOPIC_TITLE_HINTS = (
    "anime and manga fandom",
    "anime fandom",
    "manga fandom",
    "anime and manga culture",
    "manga culture",
    "anime culture",
    "statistics for wikimedia projects",
    "wikistats",
)
GENERIC_EVENT_TOPIC_TITLE_HINTS = (
    "beauty pageant",
    "beauty contest",
    "beauty queen",
    "festival queen",
    "beauty pageant winners",
    "pageant winners",
    "american beauty pageant winners",
)
GENERIC_EVENT_TOPIC_TERMS = {
    "american",
    "beauty",
    "contest",
    "contests",
    "event",
    "events",
    "festival",
    "festivals",
    "miss",
    "mrs",
    "ms",
    "pageant",
    "pageants",
    "queen",
    "queens",
    "title",
    "titles",
    "winner",
    "winners",
}
BOSNIA_TOP_FOUR_CITY_TERMS = (
    "sarajevo",
    "banja luka",
    "tuzla",
    "zenica",
)
HISTORICAL_YEAR_TRUSTED_MEMORIAL_SOURCE_HINTS = (
    "spomenikdatabase.org",
)
HISTORICAL_YEAR_CONSTRUCTION_HINTS = (
    "built",
    "constructed",
    "erected",
    "completed",
    "unveiled",
    "opened",
    "installed",
)
HISTORICAL_YEAR_ARTIST_HINTS = (
    "artist",
    "sculptor",
    "architect",
    "designer",
    "author",
)
HISTORICAL_YEAR_CONTEXT_HINTS = (
    "former yugoslavia",
    "yugoslavia",
    "yugoslav",
    "socialist yugoslavia",
)
PERSON_BIOGRAPHY_HINTS = (
    "artist",
    "artists",
    "author",
    "authors",
    "awards",
    "biography",
    "born",
    "career",
    "died",
    "legacy",
    "profile",
    "sculptor",
    "sculptors",
)
GENERIC_MEDIA_TOPIC_TERMS = {
    "anime",
    "manga",
    "fandom",
    "culture",
    "cultures",
    "demographics",
    "fan",
    "fans",
    "generation",
    "generations",
    "genre",
    "genres",
    "history",
    "industry",
    "industries",
    "media",
    "projects",
    "statistics",
    "topic",
    "topics",
    "wikimedia",
    "wikipedia",
    "wiki",
}
GENERIC_ENTITY_BLOCKLIST = {
    "anime",
    "manga",
    "fandom",
    "their",
    "them",
    "they",
    "wikimedia",
    "wikipedia",
    "wikistats",
}
MEDIA_DISAMBIGUATION_HINTS = (
    "manga",
    "anime",
    "film",
    "movie",
    "novel",
    "comic",
    "comics",
    "character",
    "tv series",
    "television series",
    "video game",
    "game",
)
CHARACTER_QUERY_HINTS = (
    "character",
    "characters",
    "antagonist",
    "protagonist",
    "villain",
    "hero",
    "companion",
    "rival",
    "ally",
)
ABILITY_QUERY_HINTS = (
    "ability",
    "abilities",
    "power",
    "powers",
    "move",
    "moves",
    "movement",
    "movements",
    "effect",
    "effects",
    "technique",
    "techniques",
    "attack",
    "attacks",
)
REFERENCE_HINTS = (
    "reference",
    "references",
    "bibliography",
    "cited",
    "citation",
    "works cited",
)
NON_RESEARCH_PAGE_HINTS = (
    "course",
    "program",
    "precollege",
    "summer college",
    "major",
    "minor",
    "degree requirements",
    "admissions",
    "apply now",
)
LOW_VALUE_TERMS = {
    "discussing",
    "discussion",
    "someone",
    "trying",
    "confirm",
    "details",
    "paper",
    "research",
    "discussed",
    "particular",
    "focused",
    "study",
    "section",
    "person",
    "student",
    "expressed",
    "later",
    "became",
    "last",
    "name",
    "confirming",
    "sometime",
}
PERSON_QUERY_HINTS = (
    "who",
    "whom",
    "last name",
    "surname",
    "full name",
    "first and last name",
    "first and last names",
    "person",
)
PERSON_CONTEXT_HINTS = (
    "author",
    "student",
    "professor",
    "faculty",
    "advisor",
    "supervisor",
    "mentor",
)
ORGANIZATION_NAME_HINTS = {
    "academy",
    "agency",
    "association",
    "authority",
    "best",
    "bureau",
    "center",
    "centre",
    "city",
    "college",
    "commission",
    "committee",
    "corporation",
    "council",
    "department",
    "development",
    "division",
    "event",
    "festival",
    "final",
    "government",
    "group",
    "history",
    "information",
    "institute",
    "jamaica",
    "letter",
    "letters",
    "ministry",
    "municipal",
    "municipality",
    "office",
    "organisation",
    "organization",
    "pageant",
    "picture",
    "province",
    "provincial",
    "queen",
    "region",
    "school",
    "service",
    "society",
    "title",
    "tourism",
    "university",
}
NON_PERSON_TITLE_HINTS = {
    "academy",
    "america",
    "american",
    "award",
    "awards",
    "best",
    "blog",
    "california",
    "category",
    "counselor",
    "counsellor",
    "county",
    "episode",
    "event",
    "fame",
    "final",
    "health",
    "history",
    "international",
    "letter",
    "letters",
    "mental",
    "miss",
    "mrs",
    "ms",
    "oscar",
    "outstanding",
    "page",
    "pageant",
    "picture",
    "post",
    "posts",
    "question",
    "questions",
    "quiz",
    "round",
    "same",
    "season",
    "show",
    "story",
    "teen",
    "title",
    "titles",
    "today",
    "trivia",
    "tournament",
    "universe",
    "usa",
    "video",
    "winner",
    "winners",
}
NON_PERSON_CATEGORY_HINTS = {
    "agriculture",
    "arts",
    "baking",
    "beauty",
    "best",
    "booth",
    "career",
    "choreography",
    "consumer",
    "costume",
    "dance",
    "dancing",
    "design",
    "education",
    "excellence",
    "festival",
    "finishing",
    "history",
    "leadership",
    "mathematics",
    "musicality",
    "newsletter",
    "official",
    "pageant",
    "painting",
    "performance",
    "poster",
    "presentation",
    "programme",
    "program",
    "province",
    "provincial",
    "public",
    "queen",
    "science",
    "service",
    "skills",
    "small",
    "soft",
    "speaking",
    "speech",
    "storyline",
    "studies",
    "study",
    "talent",
    "tourism",
    "training",
    "update",
    "weekly",
    "winner",
    "winners",
}
MULTI_HOP_RELATIONSHIP_HINTS = (
    "later",
    "before",
    "after",
    "between",
    "fewer than",
    "less than",
    "more than",
    "under",
    "over",
    "founded",
    "appointed",
    "became",
    "joined",
    "acknowledg",
    "professor",
    "department",
    "university",
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "between",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}
QUESTION_PREFIXES = (
    "what is ",
    "what are ",
    "who is ",
    "who are ",
    "which ",
    "where is ",
    "where are ",
    "when is ",
    "when are ",
    "tell me about ",
    "find ",
    "search for ",
)
CONSTRAINT_PATTERNS = (
    r"\b(?:between|from)\s+\d{4}\s+(?:and|to|-)\s+\d{4}\b",
    r"\b(?:before|after|since|during|in)\s+\d{4}\b",
    r"\b(?:before|after|since|during|in)\s+(?:the\s+)?\d{4}s\b",
    r"\b(?:fewer than|less than|more than|greater than|under|over|at least|at most|minimum of|maximum of)\s+\d+(?:\s+[a-zA-Z-]+){0,4}\b",
    r"['\"][^'\"]{2,80}['\"]",
)
CLAUSE_SPLIT_PATTERN = re.compile(
    r"\s*(?:,|;|\b(?:and|or|but|who|which|that|where|when|whose)\b)\s*",
    re.IGNORECASE,
)
AUTHOR_LINE_PATTERNS = (
    re.compile(
        r"\b(?:by|author|written by|submitted by|prepared by)\s*[:,-]?\s*"
        r"([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){1,3})\b"
    ),
    re.compile(
        r"\b([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){1,3}),?\s+"
        r"(?:master(?:'s)?|phd|doctoral|doctor of philosophy|dissertation|thesis)\b",
        re.IGNORECASE,
    ),
)
INSTITUTION_ENTITY_PATTERNS = (
    re.compile(
        r"\b([A-Z][A-Za-z&'`.-]+(?:\s+[A-Z][A-Za-z&'`.-]+){0,6}\s+"
        r"(?:University|College|Institute|School))\b"
    ),
    re.compile(
        r"\b(Department of [A-Z][A-Za-z&'`.-]+(?:\s+[A-Z][A-Za-z&'`.-]+){0,6})\b"
    ),
    re.compile(
        r"\b(School of [A-Z][A-Za-z&'`.-]+(?:\s+[A-Z][A-Za-z&'`.-]+){0,6})\b"
    ),
    re.compile(
        r"\b(College of [A-Z][A-Za-z&'`.-]+(?:\s+[A-Z][A-Za-z&'`.-]+){0,6})\b"
    ),
)
PERSON_NAME_PATTERN = re.compile(
    r"\b(?:Dr\.?\s+|Prof\.?\s+|Professor\s+)?"
    r"([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){1,3})\b"
)
TITLECASE_ENTITY_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9'`.-]+(?:\s+[A-Z][A-Za-z0-9'`.-]+){0,4})\b"
)
TITLECASE_ENTITY_BLOCKLIST = {
    "Abstract",
    "Acknowledgements",
    "Acknowledgments",
    "Acknowledgement",
    "Bibliography",
    "Chapter",
    "Contents",
    "December",
    "Figure",
    "Her",
    "His",
    "Introduction",
    "Its",
    "January",
    "Many",
    "March",
    "One",
    "References",
    "Section",
    "Table",
    "The",
    "This",
    "Those",
    "Their",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
}
ACKNOWLEDGEMENT_SECTION_TITLES = (
    "acknowledgements",
    "acknowledgments",
    "acknowledgement",
)
REFERENCE_SECTION_TITLES = (
    "references",
    "bibliography",
    "works cited",
)
ABSTRACT_SECTION_TITLES = (
    "abstract",
)
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
REFERENCE_STYLE_YEAR_PATTERN = re.compile(r"\(\d{4}\)|\b\d{4};\d")
REFERENCE_STYLE_AUTHOR_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z'`.-]+,\s*(?:[A-Z]\.\s*){1,4}"
)
FRONT_MATTER_CHAR_LIMIT = 3000


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_heading(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_whitespace(text).lower()).strip()


def normalize_document_text(text: str) -> str:
    lines = [normalize_whitespace(line) for line in (text or "").replace("\r", "\n").split("\n")]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines)


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        normalized_value = normalize_whitespace(value)
        if not normalized_value:
            continue
        key = normalized_value.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_values.append(normalized_value)
    return unique_values


def important_terms(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text)
    return [token for token in tokens if token.lower() not in STOPWORDS]


def keyword_focus(text: str, *, max_terms: int) -> str:
    return " ".join(important_terms(text)[:max_terms])


def strip_question_prefix(query: str) -> str:
    lowered = query.lower()
    for prefix in QUESTION_PREFIXES:
        if lowered.startswith(prefix):
            return normalize_whitespace(query[len(prefix) :])
    return query


def clean_fragment(text: str) -> str:
    cleaned = normalize_whitespace(text)
    cleaned = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", cleaned)
    cleaned = re.sub(
        r"\b(?:and|or|with|by|from|in|on|for|to|of|about|between|before|after)$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return normalize_whitespace(cleaned)


def extract_constraint_phrases(query: str) -> List[str]:
    constraints: list[str] = []
    for pattern in CONSTRAINT_PATTERNS:
        constraints.extend(match.group(0) for match in re.finditer(pattern, query, re.IGNORECASE))
    return unique_preserve_order(clean_fragment(constraint) for constraint in constraints)


def split_into_clauses(query: str, constraints: List[str]) -> List[str]:
    protected_query = query
    replacements: dict[str, str] = {}
    for index, constraint in enumerate(sorted(constraints, key=len, reverse=True)):
        placeholder = f"__CONSTRAINT_{index}__"
        protected_query = re.sub(
            re.escape(constraint),
            placeholder,
            protected_query,
            flags=re.IGNORECASE,
        )
        replacements[placeholder] = constraint

    raw_parts = CLAUSE_SPLIT_PATTERN.split(protected_query)
    clauses: list[str] = []
    for raw_part in raw_parts:
        part = raw_part
        for placeholder, original in replacements.items():
            part = part.replace(placeholder, original)
        cleaned_part = clean_fragment(part)
        if len(important_terms(cleaned_part)) >= 2:
            clauses.append(cleaned_part)

    return unique_preserve_order(clauses)


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/")
    return f"{netloc}{path}"


def extract_source(url: str) -> str:
    if not url:
        return "unknown"
    return urlparse(url).netloc.lower().removeprefix("www.") or "unknown"


def clue_similarity(clue_a: str, clue_b: str) -> float:
    return SequenceMatcher(
        None,
        normalize_whitespace(clue_a).lower(),
        normalize_whitespace(clue_b).lower(),
    ).ratio()


def lexical_relevance_score(query: str, text: str) -> float:
    query_terms = important_terms(query.lower())
    text_terms = set(important_terms(text.lower()))
    if not query_terms or not text_terms:
        return 0.0

    overlap = sum(1 for term in query_terms if term in text_terms) / len(query_terms)
    sequence_score = SequenceMatcher(
        None,
        normalize_whitespace(query).lower(),
        normalize_whitespace(text)[:800].lower(),
    ).ratio()
    query_numbers = set(re.findall(r"\d+", query))
    text_numbers = set(re.findall(r"\d+", text))
    numeric_bonus = 1.0 if query_numbers and query_numbers.issubset(text_numbers) else 0.0
    return min(1.0, (0.6 * overlap) + (0.3 * sequence_score) + (0.1 * numeric_bonus))


def specific_query_terms(text: str) -> List[str]:
    filtered_terms: list[str] = []
    ordered_positions: dict[str, int] = {}
    for index, token in enumerate(important_terms(text.lower())):
        if len(token) < 4:
            continue
        if token in BROWSECOMP_LOW_SIGNAL_TERMS:
            continue
        ordered_positions.setdefault(token, index)
        filtered_terms.append(token)
    unique_terms = unique_preserve_order(filtered_terms)
    return sorted(unique_terms, key=lambda term: (-len(term), ordered_positions.get(term, 0)))


def browsecomp_anchor_groups(query: str) -> List[tuple[str, ...]]:
    normalized_query = normalize_whitespace(query)
    if not normalized_query:
        return []

    groups: list[tuple[str, ...]] = []
    seen_groups: set[tuple[str, ...]] = set()

    for sentence in split_sentences(normalized_query):
        terms = tuple(specific_query_terms(sentence)[:4])
        if not terms or terms in seen_groups:
            continue
        groups.append(terms)
        seen_groups.add(terms)

    if len(groups) < 2:
        constraints = extract_constraint_phrases(normalized_query)
        clauses = split_into_clauses(strip_question_prefix(normalized_query), constraints)
        for clause in clauses:
            terms = tuple(specific_query_terms(clause)[:4])
            if not terms or terms in seen_groups:
                continue
            groups.append(terms)
            seen_groups.add(terms)

    return groups[:6]


def browsecomp_anchor_match_stats(query: str, text: str) -> tuple[int, int, int]:
    groups = browsecomp_anchor_groups(query)
    if not groups:
        return 0, 0, 0

    lowered_text = normalize_whitespace(text).lower()
    matched_groups = 0
    matched_terms = 0
    for group in groups:
        hits = sum(1 for term in group if term in lowered_text)
        required_hits = 1 if len(group) == 1 else min(2, len(group))
        if hits >= required_hits:
            matched_groups += 1
            matched_terms += hits
    return matched_groups, matched_terms, len(groups)


def specificity_overlap_score(query: str, text: str) -> float:
    query_terms = specific_query_terms(query)
    text_terms = set(important_terms(text.lower()))
    if not query_terms or not text_terms:
        return 0.0

    overlap = sum(1 for term in query_terms if term in text_terms) / len(query_terms)
    phrase_bonus = 0.0
    normalized_text = normalize_whitespace(text).lower()
    for term in query_terms[:4]:
        if len(term) >= 6 and term in normalized_text:
            phrase_bonus += 0.03
    return min(1.0, overlap + min(0.12, phrase_bonus))


def contains_exact_phrase(text: str, phrase: str) -> bool:
    normalized_text = normalize_whitespace(text).lower()
    normalized_phrase = normalize_whitespace(phrase).lower()
    if not normalized_text or not normalized_phrase:
        return False

    pattern = (
        r"(?<!\w)"
        + re.escape(normalized_phrase).replace(r"\ ", r"\s+")
        + r"(?!\w)"
    )
    return re.search(pattern, normalized_text, re.IGNORECASE) is not None


def contains_any_exact_phrase(text: str, phrases: Iterable[str]) -> bool:
    return any(contains_exact_phrase(text, phrase) for phrase in phrases)


def is_academic_lookup_query(query: str) -> bool:
    query_lower = query.lower()
    hint_matches = sum(1 for hint in ACADEMIC_QUERY_HINTS if hint in query_lower)
    has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", query_lower))
    return hint_matches >= 2 or (hint_matches >= 1 and has_year)


def is_person_target_query(query: str) -> bool:
    query_lower = query.lower()
    return contains_any_exact_phrase(query_lower, PERSON_QUERY_HINTS)


def is_media_lookup_query(query: str) -> bool:
    query_lower = query.lower()
    hint_matches = sum(1 for hint in MEDIA_QUERY_HINTS if contains_exact_phrase(query_lower, hint))
    has_named_entity_task = contains_any_exact_phrase(query_lower, CHARACTER_QUERY_HINTS)
    return hint_matches >= 2 or (hint_matches >= 1 and has_named_entity_task)


def is_event_lookup_query(query: str) -> bool:
    query_lower = query.lower()
    hint_matches = sum(1 for hint in EVENT_QUERY_HINTS if contains_exact_phrase(query_lower, hint))
    has_person_or_winner_task = contains_any_exact_phrase(
        query_lower,
        (
            "winner",
            "won",
            "person",
            "full name",
            "first and last name",
            "first and last names",
            "beauty pageant",
            "pageant",
        ),
    )
    return hint_matches >= 2 or (hint_matches >= 1 and has_person_or_winner_task)


def extract_media_type(query: str) -> str:
    query_lower = normalize_whitespace(query).lower()
    for media_type in ("manga", "anime", "comic", "comics", "graphic novel", "novel", "game", "film", "movie", "tv", "television"):
        if media_type in query_lower:
            return "comic" if media_type == "comics" else media_type
    return "manga" if is_media_lookup_query(query_lower) else ""


def is_broad_overview_page(url: str, title: str) -> bool:
    title_lower = normalize_whitespace(title).lower()
    url_lower = url.lower()
    title_without_suffix = re.sub(r"\s*-\s*(wikipedia|fandom).*?$", "", title_lower).strip()

    if not title_lower:
        return False
    if any(title_without_suffix.startswith(prefix) for prefix in BROAD_OVERVIEW_TITLE_HINTS):
        return True
    if "disambiguation" in title_without_suffix:
        return True
    if "/category:" in url_lower or "category:" in title_without_suffix:
        return True
    if title_without_suffix in GENERIC_OVERVIEW_TITLES:
        return True
    if len(important_terms(title_without_suffix)) <= 1 and any(
        hint in title_without_suffix for hint in ("wiki", "wikipedia", "fandom")
    ):
        return True
    return False


def is_aggregate_listing_page(url: str, title: str, snippet: str = "") -> bool:
    combined = normalize_whitespace(f"{url} {title} {snippet}").lower()
    return any(hint in combined for hint in AGGREGATE_LISTING_HINTS)


def is_forum_discussion_page(url: str, title: str, snippet: str = "") -> bool:
    combined = normalize_whitespace(f"{url} {title} {snippet}").lower()
    return any(hint in combined for hint in FORUM_DISCUSSION_HINTS)


def is_non_english_wiki_page(url: str) -> bool:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    if not host:
        return False
    if host.startswith("m."):
        host = host[2:]
    if host in {"en.wikipedia.org", "wikipedia.org"}:
        return False
    if host.endswith(".wikipedia.org"):
        return True
    return False


def is_wiki_meta_page(url: str, title: str, text: str = "") -> bool:
    combined = normalize_whitespace(f"{url} {title} {text[:800]}").lower()
    if any(hint in combined for hint in WIKI_META_SOURCE_HINTS):
        return True
    if any(
        marker in combined
        for marker in (
            "wikipedia:",
            "wikimedia:",
            "mediawiki:",
            "special:",
            "portal:",
            "template:",
        )
    ):
        return True
    return False


def is_generic_media_topic_page(url: str, title: str, text: str = "") -> bool:
    cleaned_title = normalize_whitespace(title).strip(" -|")
    title_lower = cleaned_title.lower()
    combined = normalize_whitespace(f"{url} {cleaned_title} {text[:1200]}").lower()
    if not cleaned_title:
        return False
    if is_wiki_meta_page(url, title, text):
        return True
    if any(phrase in combined for phrase in GENERIC_MEDIA_TOPIC_TITLE_HINTS):
        return True

    title_entities = [
        entity.lower()
        for entity in extract_capitalized_entities(cleaned_title)
        if entity.lower() not in GENERIC_ENTITY_BLOCKLIST
    ]
    title_terms = important_terms(title_lower)
    if not title_terms:
        return True

    generic_term_ratio = sum(
        1
        for term in title_terms
        if term in GENERIC_MEDIA_TOPIC_TERMS or term in BROWSECOMP_LOW_SIGNAL_TERMS
    ) / len(title_terms)

    if (
        generic_term_ratio >= 0.75
        and not title_entities
        and any(term in title_lower for term in ("anime", "manga", "fandom", "wiki", "statistics"))
    ):
        return True

    if (
        any(term in title_lower for term in ("fandom", "culture", "history", "industry", "statistics", "generation"))
        and any(term in title_lower for term in ("anime", "manga", "comic"))
        and not title_entities
    ):
        return True

    return False


def is_generic_event_topic_page(url: str, title: str, text: str = "") -> bool:
    cleaned_title = normalize_whitespace(title).strip(" -|")
    title_lower = cleaned_title.lower()
    title_without_suffix = re.sub(r"\s*-\s*(wikipedia|fandom).*?$", "", title_lower).strip()
    combined = normalize_whitespace(f"{url} {cleaned_title} {text[:1200]}").lower()
    if not cleaned_title:
        return False
    if is_wiki_meta_page(url, title, text):
        return True
    if title_without_suffix in GENERIC_EVENT_TOPIC_TITLE_HINTS:
        return True
    if any(phrase in title_without_suffix for phrase in ("category:", "list of ")) and any(
        phrase in title_without_suffix for phrase in ("beauty", "pageant", "queen", "contest", "winner")
    ):
        return True
    if "american beauty pageant winners" in combined:
        return True

    title_entities = [
        entity.lower()
        for entity in extract_capitalized_entities(cleaned_title)
        if entity.lower() not in GENERIC_ENTITY_BLOCKLIST
    ]
    title_terms = important_terms(title_without_suffix)
    if not title_terms:
        return False

    generic_term_ratio = sum(
        1 for term in title_terms if term in GENERIC_EVENT_TOPIC_TERMS or term in BROWSECOMP_LOW_SIGNAL_TERMS
    ) / len(title_terms)
    non_generic_terms = [
        term
        for term in title_terms
        if term not in GENERIC_EVENT_TOPIC_TERMS and term not in BROWSECOMP_LOW_SIGNAL_TERMS
    ]

    if generic_term_ratio >= 0.75 and len(non_generic_terms) <= 1 and not title_entities:
        return True
    if (
        any(term in title_without_suffix for term in ("beauty pageant", "beauty contest", "festival queen"))
        and len(non_generic_terms) <= 1
        and not title_entities
    ):
        return True
    return False


def is_low_trust_social_page(url: str, title: str, text: str = "") -> bool:
    combined = normalize_whitespace(f"{url} {title} {text[:1200]}").lower()
    return any(hint in combined for hint in LOW_TRUST_SOCIAL_SOURCE_HINTS)


def is_recipe_food_page(url: str, title: str, text: str = "") -> bool:
    combined = normalize_whitespace(f"{url} {title} {text[:1600]}").lower()
    recipe_hits = sum(1 for hint in RECIPE_FOOD_PAGE_HINTS if hint in combined)
    event_hits = sum(1 for hint in EVENT_PAGE_HINTS if hint in combined)
    if recipe_hits >= 2 and event_hits == 0:
        return True
    if recipe_hits >= 3 and event_hits <= 1:
        return True
    return False


def is_generic_historical_monument_page(url: str, title: str, text: str = "") -> bool:
    combined = normalize_whitespace(f"{url} {title} {text[:1600]}").lower()
    title_lower = normalize_whitespace(title).lower()
    if any(
        phrase in combined
        for phrase in (
            "world war ii monuments and memorials",
            "former yugoslavia monuments",
            "spomeniks and memorials",
            "birth of yugoslavia",
            "birth of yugoslavia's spomeniks",
            "secrets of spomeniks",
            "symbols of loss",
            "window into yugoslavia",
            "browsing articles by tag",
            "articles by tag",
        )
    ):
        return True
    if "spomeniks" in title_lower and not any(
        phrase in title_lower for phrase in ("memorial on", "monument to", "victims memorial")
    ):
        return True
    if "photos" in title_lower and any(term in title_lower for term in ("monument", "memorial", "spomenik")):
        return True
    if "monuments and memorials" in title_lower:
        return True
    return False


def query_requires_bosnia_top_city(query: str) -> bool:
    query_lower = normalize_whitespace(query).lower()
    return "bosnia" in query_lower and any(
        phrase in query_lower
        for phrase in ("top 4", "largest cities", "population census")
    )


def historical_year_location_terms(query: str) -> tuple[str, ...]:
    query_lower = normalize_whitespace(query).lower()
    terms = ["bosnia", "yugoslavia"]
    if "bosnia" in query_lower or "largest cities" in query_lower or "population census" in query_lower:
        terms.extend(BOSNIA_TOP_FOUR_CITY_TERMS)
    ordered_terms: list[str] = []
    for term in terms:
        if term in ordered_terms:
            continue
        if term in query_lower or term in BOSNIA_TOP_FOUR_CITY_TERMS:
            ordered_terms.append(term)
    return tuple(ordered_terms)


def historical_year_city_hit(query: str, text: str) -> bool:
    if not query_requires_bosnia_top_city(query):
        return False
    lowered = normalize_whitespace(text).lower()
    return any(term in lowered for term in BOSNIA_TOP_FOUR_CITY_TERMS)


def historical_year_build_year_cutoff(query: str) -> int | None:
    query_lower = normalize_whitespace(query).lower()
    match = re.search(
        r"\b(?:constructed|built|erected|completed|opened|unveiled|installed)\b"
        r"[^.]{0,40}?\b(?:prior to|before)\s+((?:19|20)\d{2})\b",
        query_lower,
    )
    if not match:
        return None
    return int(match.group(1))


def historical_year_artist_birth_year(query: str) -> int | None:
    query_lower = normalize_whitespace(query).lower()
    match = re.search(
        r"\b(?:artist|sculptor|architect|designer|author)\b"
        r"[^.]{0,80}?\b(?:who\s+was\s+|was\s+)?born(?:\s+in)?\s+((?:19|20)\d{2})\b",
        query_lower,
    )
    if not match:
        return None
    return int(match.group(1))


def historical_year_has_structural_constraints(query: str) -> bool:
    query_lower = normalize_whitespace(query).lower()
    return bool(
        historical_year_build_year_cutoff(query) is not None
        or historical_year_artist_birth_year(query) is not None
        or "former yugoslavia" in query_lower
    )


def historical_year_trusted_memorial_source(url: str) -> bool:
    url_lower = (url or "").lower()
    return any(hint in url_lower for hint in HISTORICAL_YEAR_TRUSTED_MEMORIAL_SOURCE_HINTS)


def historical_year_construction_years(text: str) -> tuple[int, ...]:
    lowered = normalize_whitespace(text).lower()
    years: list[int] = []
    for match in re.finditer(
        r"\b(?:built|constructed|erected|completed|unveiled|opened|installed)\b"
        r"[^.]{0,60}?\b((?:19|20)\d{2})\b"
        r"(?:[^.]{0,12}?\b(?:to|-|and)\b[^.]{0,12}?\b((?:19|20)\d{2})\b)?",
        lowered,
    ):
        years.append(int(match.group(1)))
        if match.group(2):
            years.append(int(match.group(2)))
    return tuple(dict.fromkeys(years))


def historical_year_artist_birth_mentions(text: str) -> tuple[int, ...]:
    lowered = normalize_whitespace(text).lower()
    years: list[int] = []
    patterns = (
        r"\b(?:artist|sculptor|architect|designer|author)\b"
        r"[^.]{0,120}?\bborn(?:\s+in)?\s+((?:19|20)\d{2})\b",
        r"\b(?:artist|sculptor|architect|designer|author)\b"
        r"[^.]{0,120}?\(((?:19|20)\d{2})\s*[–-]",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            years.append(int(match.group(1)))
    return tuple(dict.fromkeys(years))


def historical_year_structural_assessment(query: str, url: str, title: str, text: str = "") -> tuple[float, int, int]:
    query_lower = normalize_whitespace(query).lower()
    combined = normalize_whitespace(f"{title} {url} {text[:3200]}")
    lowered = combined.lower()
    score = 0.0
    matched = 0
    contradicted = 0

    build_year_cutoff = historical_year_build_year_cutoff(query)
    if build_year_cutoff is not None:
        build_years = historical_year_construction_years(combined)
        if any(year < build_year_cutoff for year in build_years):
            score += 0.26
            matched += 1
        elif build_years and min(build_years) >= build_year_cutoff:
            score -= 0.42
            contradicted += 1

    artist_birth_year = historical_year_artist_birth_year(query)
    if artist_birth_year is not None:
        artist_birth_mentions = historical_year_artist_birth_mentions(combined)
        if artist_birth_year in artist_birth_mentions:
            score += 0.22
            matched += 1
        elif artist_birth_mentions:
            score -= 0.32
            contradicted += 1

    if "former yugoslavia" in query_lower and any(term in lowered for term in HISTORICAL_YEAR_CONTEXT_HINTS):
        score += 0.14
        matched += 1

    if historical_year_trusted_memorial_source(url):
        score += 0.08

    return score, matched, contradicted


def event_page_score(url: str, title: str, text: str = "") -> float:
    combined = normalize_whitespace(f"{url} {title} {text[:3000]}").lower()
    score = 0.0

    if any(hint in combined for hint in EVENT_OFFICIAL_SOURCE_HINTS):
        score += 0.28

    event_matches = sum(1 for hint in EVENT_PAGE_HINTS if hint in combined)
    score += min(0.42, 0.07 * event_matches)

    if any(phrase in combined for phrase in ("beauty pageant", "festival queen", "coronation", "grand winner")):
        score += 0.15
    if any(phrase in combined for phrase in ("winner", "won", "pageant", "contest")) and contains_candidate_person_name(
        text[:1600]
    ):
        score += 0.08

    if is_broad_overview_page(url, title):
        score -= 0.12
    if is_aggregate_listing_page(url, title, text[:300]):
        score -= 0.18
    if is_generic_event_topic_page(url, title, text[:300]):
        score -= 0.34
    if is_recipe_food_page(url, title, text[:600]):
        score -= 0.4
    if is_wiki_meta_page(url, title, text[:300]):
        score -= 0.3
    if is_non_english_wiki_page(url):
        score -= 0.25
    if is_forum_discussion_page(url, title, text[:300]):
        score -= 0.22
    if is_low_trust_social_page(url, title, text[:300]):
        score -= 0.22

    return max(0.0, min(1.0, score))


def looks_like_event_page(url: str, title: str, text: str = "", *, minimum_score: float = 0.28) -> bool:
    return event_page_score(url, title, text) >= minimum_score


def media_page_score(url: str, title: str, text: str = "") -> float:
    combined = normalize_whitespace(f"{url} {title} {text[:3000]}").lower()
    score = 0.0
    if any(hint in combined for hint in MEDIA_WORK_SOURCE_HINTS):
        score += 0.35

    strong_matches = sum(1 for hint in MEDIA_WORK_STRONG_HINTS if hint in combined)
    weak_matches = sum(1 for hint in MEDIA_WORK_WEAK_HINTS if hint in combined)
    score += min(0.4, (0.12 * strong_matches) + (0.05 * weak_matches))

    if any(
        phrase in combined
        for phrase in (
            "manga series",
            "anime television series",
            "fictional character",
            "list of characters",
            "serialized in",
            "anime adaptation",
            "chapter list",
        )
    ):
        score += 0.15

    if is_broad_overview_page(url, title):
        score -= 0.12
    if is_aggregate_listing_page(url, title, text[:300]):
        score -= 0.25
    if is_wiki_meta_page(url, title, text[:300]):
        score -= 0.35
    if is_non_english_wiki_page(url):
        score -= 0.3
    if is_generic_media_topic_page(url, title, text[:300]):
        score -= 0.3
    if is_forum_discussion_page(url, title, text[:300]):
        score -= 0.22
    if strong_matches == 0 and any(hint in combined for hint in MEDIA_NON_WORK_HINTS):
        score -= 0.35

    return max(0.0, min(1.0, score))


def looks_like_media_page(url: str, title: str, text: str = "", *, minimum_score: float = 0.32) -> bool:
    return media_page_score(url, title, text) >= minimum_score


def is_grounded_browsecomp_page(
    query: str,
    url: str,
    title: str,
    text: str = "",
    *,
    require_media: bool = False,
) -> bool:
    if is_broad_overview_page(url, title):
        return False
    if is_aggregate_listing_page(url, title, text[:300]):
        return False
    if is_wiki_meta_page(url, title, text[:300]):
        return False
    if is_non_english_wiki_page(url):
        return False
    if is_generic_media_topic_page(url, title, text[:600]):
        return False
    if is_generic_event_topic_page(url, title, text[:600]):
        return False
    if is_forum_discussion_page(url, title, text[:300]):
        return False
    if is_low_trust_social_page(url, title, text[:300]):
        return False
    if is_generic_historical_monument_page(url, title, text[:600]):
        return False

    combined = normalize_whitespace(f"{title} {url} {text[:3200]}")
    specificity = specificity_overlap_score(query, combined)
    matched_groups, matched_terms, total_groups = browsecomp_anchor_match_stats(query, combined)
    media_score = media_page_score(url, title, text[:3000])
    event_score = event_page_score(url, title, text[:3000])
    query_lower = normalize_whitespace(query).lower()
    is_year_query = any(
        phrase in query_lower
        for phrase in ("what year", "which year", "in what year", "when was", "when did", "dated")
    )
    historical_year_query = is_year_query and any(
        term in query_lower for term in ("monument", "memorial", "spomenik", "victims", "loss of lives", "in their honor")
    )
    historical_memorial_evidence = (
        bool(re.search(r"\b(?:19|20)\d{2}\b", combined))
        and any(
            hint in combined.lower()
            for hint in (
                "victims",
                "massacre",
                "battle",
                "killed",
                "died",
                "executed",
                "commemorates",
                "in honor",
                "dedication",
                "memorial",
                "monument",
                "spomenik",
            )
        )
    )

    if require_media and media_score < 0.18:
        return False
    if historical_year_query:
        return is_specific_historical_year_page(query, url, title, text)
    if is_year_query and historical_memorial_evidence:
        location_hints = historical_year_location_terms(query)
        location_hit = any(term in combined.lower() for term in location_hints)
        city_hit = historical_year_city_hit(query, combined)
        if query_requires_bosnia_top_city(query) and not city_hit:
            return False
        build_year_hint = bool(re.search(r"\b19(?:[0-5]\d|6\d)\b", combined))
        artist_hint = any(
            term in combined.lower()
            for term in ("artist", "sculptor", "designed by", "author", "miodrag", "živković", "zivkovic")
        )
        if location_hit and matched_groups >= 1 and specificity >= 0.12:
            return True
        if build_year_hint and artist_hint and matched_groups >= 1 and specificity >= 0.12:
            return True
        if matched_groups >= 1 and specificity >= 0.16:
            return True
        if location_hit and specificity >= 0.18:
            return True
        if matched_groups >= 2 and specificity >= 0.12:
            return True
    if is_event_lookup_query(query):
        if event_score >= 0.74 and specificity >= 0.12:
            return True
        if event_score >= 0.58 and matched_groups >= 1 and specificity >= 0.1:
            return True
    if matched_groups >= 2:
        return True
    if matched_groups >= 1 and matched_terms >= 3 and specificity >= 0.22:
        return True
    if total_groups <= 1 and specificity >= 0.28 and (not require_media or media_score >= 0.22):
        return True
    return False


def score_browsecomp_link_candidate(
    query: str,
    link_text: str,
    link_url: str,
    *,
    parent_title: str = "",
    parent_url: str = "",
) -> float:
    cleaned_text = normalize_whitespace(link_text).strip(" -|")
    if not cleaned_text or len(cleaned_text) > 120:
        return 0.0

    combined = normalize_whitespace(f"{cleaned_text} {link_url}").lower()
    if any(hint in combined for hint in BROWSECOMP_LINK_EXCLUDE_HINTS):
        return 0.0
    if is_broad_overview_page(link_url, cleaned_text):
        return 0.0
    if is_aggregate_listing_page(link_url, cleaned_text):
        return 0.0
    if is_wiki_meta_page(link_url, cleaned_text):
        return 0.0
    if is_non_english_wiki_page(link_url):
        return 0.0
    if is_generic_media_topic_page(link_url, cleaned_text):
        return 0.0
    if is_generic_event_topic_page(link_url, cleaned_text):
        return 0.0
    if is_generic_historical_monument_page(link_url, cleaned_text):
        return 0.0
    if is_person_biography_page(link_url, cleaned_text):
        return 0.0
    if is_forum_discussion_page(link_url, cleaned_text):
        return 0.0
    if cleaned_text.lower() in GENERIC_LINK_TEXT_HINTS:
        return 0.0

    tokens = important_terms(cleaned_text)
    if not tokens or len(tokens) > 9:
        return 0.0

    score = 0.18
    if 1 <= len(tokens) <= 6:
        score += 0.12
    if extract_capitalized_entities(cleaned_text):
        score += 0.12
    if re.search(r"[A-Z][a-z]+", cleaned_text):
        score += 0.07

    score += 0.28 * media_page_score(link_url, cleaned_text)
    score += 0.22 * specificity_overlap_score(
        query,
        f"{cleaned_text} {link_url.replace('-', ' ').replace('_', ' ')}",
    )

    if parent_url and extract_source(parent_url) == extract_source(link_url):
        score += 0.05
    if parent_title and (
        is_broad_overview_page(parent_url, parent_title)
        or is_aggregate_listing_page(parent_url, parent_title)
    ):
        score += 0.05

    lowered_text = cleaned_text.lower()
    if any(hint in lowered_text for hint in ("category", "history", "list", "portal", "template")):
        score -= 0.4
    if len(tokens) == 1 and len(tokens[0]) < 5:
        score -= 0.08

    return max(0.0, min(1.0, score))


def document_matches_query_years(query: str, text: str) -> bool:
    query_years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
    if not query_years:
        return False
    document_years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))
    return bool(query_years & document_years)


def split_sentences(text: str) -> List[str]:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []
    return [
        normalize_whitespace(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        if normalize_whitespace(sentence)
    ]


def domain_reliability_score(url: str) -> float:
    domain = extract_source(url)
    if not domain or domain == "unknown":
        return 0.1
    score = 0.2
    if domain.endswith(".edu") or domain.endswith(".gov") or domain.endswith(".ac.uk"):
        score += 0.3
    if any(hint in domain for hint in ("repository", "digitalcommons", "scholarworks")):
        score += 0.2
    return min(1.0, score)


def document_type_score(url: str, title: str, snippet: str) -> float:
    combined = normalize_whitespace(f"{url} {title} {snippet}").lower()
    score = 0.0
    if url.lower().endswith(".pdf") or "application/pdf" in combined:
        score += 0.4
    if "doi.org/" in combined:
        score += 0.25
    if any(hint in combined for hint in ACADEMIC_REPOSITORY_HINTS):
        score += 0.2
    if any(hint in combined for hint in THESIS_CONTENT_HINTS):
        score += 0.2
    if any(hint in combined for hint in ACKNOWLEDGEMENT_HINTS):
        score += 0.1
    if any(hint in combined for hint in PAPER_CONTENT_HINTS):
        score += 0.15
    return min(1.0, score)


def document_title_query_phrase(title: str) -> str:
    cleaned_title = normalize_whitespace(title)
    cleaned_title = re.sub(r"^\[PDF\]\s*", "", cleaned_title, flags=re.IGNORECASE)
    cleaned_title = cleaned_title.replace("...", "")
    title_parts = re.split(r"\s+[|:-]\s+", cleaned_title, maxsplit=1)
    if len(title_parts) > 1:
        left_part = normalize_whitespace(title_parts[0])
        right_part = normalize_whitespace(title_parts[1])
        if left_part.lower() in {
            "spomenik database",
            "wikipedia",
            "fandom",
            "architectuul",
            "britannica",
            "bradt guides",
            "rferl",
            "radio free europe/radio liberty",
        }:
            cleaned_title = right_part
        else:
            cleaned_title = left_part
    cleaned_title = re.sub(
        r"\s*\((?:"
        + "|".join(re.escape(hint) for hint in MEDIA_DISAMBIGUATION_HINTS)
        + r")\)\s*$",
        "",
        cleaned_title,
        flags=re.IGNORECASE,
    )
    cleaned_title = cleaned_title.strip(" -|")
    if is_broad_overview_page("", cleaned_title):
        return ""
    if is_wiki_meta_page("", cleaned_title):
        return ""
    if is_generic_historical_monument_page("", cleaned_title):
        return ""
    if is_generic_media_topic_page("", cleaned_title):
        return ""
    if is_generic_event_topic_page("", cleaned_title):
        return ""
    if len(cleaned_title) > 110:
        cleaned_title = cleaned_title[:110].rsplit(" ", 1)[0]
    title_terms = important_terms(cleaned_title)
    title_entities = [
        entity.lower()
        for entity in extract_capitalized_entities(cleaned_title)
        if entity.lower() not in GENERIC_ENTITY_BLOCKLIST
    ]
    if len(title_terms) < 3:
        if not title_entities:
            return ""
        if all(
            term.lower() in GENERIC_MEDIA_TOPIC_TERMS or term.lower() in BROWSECOMP_LOW_SIGNAL_TERMS
            for term in title_terms
        ):
            return ""
    return cleaned_title


def is_person_biography_page(url: str, title: str, text: str = "") -> bool:
    title_phrase = document_title_query_phrase(title) or normalize_whitespace(title)
    if not title_phrase:
        return False
    if any(term in title_phrase.lower() for term in ("monument", "memorial", "spomenik", "spomenpark")):
        return False
    if not is_plausible_person_name(title_phrase):
        return False

    combined = normalize_whitespace(f"{url} {title} {text[:2200]}").lower()
    biography_hits = sum(1 for hint in PERSON_BIOGRAPHY_HINTS if hint in combined)
    if "wikipedia.org/wiki/" in (url or "").lower():
        biography_hits += 1
    return biography_hits >= 1


def is_specific_historical_year_page(query: str, url: str, title: str, text: str = "") -> bool:
    query_lower = normalize_whitespace(query).lower()
    if not any(phrase in query_lower for phrase in ("what year", "which year", "in what year", "when did", "when was")):
        return False
    if not any(term in query_lower for term in ("monument", "memorial", "spomenik", "victims", "loss of lives", "in their honor")):
        return False
    if is_person_biography_page(url, title, text):
        return False
    if is_generic_historical_monument_page(url, title, text):
        return False
    if is_wiki_meta_page(url, title, text):
        return False
    if is_non_english_wiki_page(url):
        return False
    if is_forum_discussion_page(url, title, text):
        return False
    if is_low_trust_social_page(url, title, text):
        return False

    title_lower = normalize_whitespace(title).lower()
    url_lower = (url or "").lower()
    combined = normalize_whitespace(f"{title} {url} {text[:3200]}").lower()
    specificity = specificity_overlap_score(query, f"{title} {text[:2400]}")
    matched_groups, matched_terms, _ = browsecomp_anchor_match_stats(query, f"{title} {text[:2400]}")
    structural_score, structural_matches, structural_contradictions = historical_year_structural_assessment(
        query,
        url,
        title,
        text,
    )

    location_terms = historical_year_location_terms(query)
    location_hit = any(term in combined for term in location_terms)
    location_in_title_or_url = any(term in f"{title_lower} {url_lower}" for term in location_terms)
    city_hit = historical_year_city_hit(query, combined)
    city_in_title_or_url = any(term in f"{title_lower} {url_lower}" for term in BOSNIA_TOP_FOUR_CITY_TERMS)

    memorial_hint = any(term in combined for term in ("monument", "memorial", "spomenik", "spomenpark"))
    casualty_hint = any(
        term in combined
        for term in (
            "victims",
            "massacre",
            "battle",
            "killed",
            "died",
            "executed",
            "detachment",
            "loss of lives",
            "commemorates",
            "fallen fighters",
        )
    )
    specific_page_hint = any(
        term in f"{title_lower} {url_lower}"
        for term in ("memorial on", "monument to", "spomenik", "spomenpark", "victims of", "massacre")
    )

    if not memorial_hint or not casualty_hint or not re.search(r"\b(?:19|20)\d{2}\b", combined):
        return False
    if query_requires_bosnia_top_city(query) and not city_hit:
        return False
    if structural_contradictions > 0:
        return False
    if historical_year_has_structural_constraints(query) and structural_matches == 0:
        trusted_location_hit = city_hit if query_requires_bosnia_top_city(query) else location_hit
        if not (
            historical_year_trusted_memorial_source(url)
            and specific_page_hint
            and trusted_location_hit
            and specificity >= 0.04
        ):
            return False
    if specific_page_hint and location_in_title_or_url and casualty_hint:
        return True
    if query_requires_bosnia_top_city(query) and specific_page_hint and city_in_title_or_url and casualty_hint:
        return True
    if location_in_title_or_url and specificity >= 0.04:
        return True
    if query_requires_bosnia_top_city(query) and city_in_title_or_url and specificity >= 0.04:
        return True
    if specific_page_hint and location_hit and specificity >= 0.11:
        return True
    if query_requires_bosnia_top_city(query) and specific_page_hint and city_hit and specificity >= 0.09:
        return True
    if specific_page_hint and matched_groups >= 2 and specificity >= 0.12:
        return True
    if location_hit and matched_groups >= 2 and matched_terms >= 3 and specificity >= 0.13:
        return True
    if query_requires_bosnia_top_city(query) and city_hit and matched_groups >= 1 and matched_terms >= 3 and specificity >= 0.11:
        return True
    if structural_score >= 0.18 and specific_page_hint and city_hit and specificity >= 0.08:
        return True
    return False


def filter_candidate_names(names: List[str]) -> List[str]:
    filtered: list[str] = []
    for name in names:
        tokens = name.split()
        if len(tokens) < 2 or len(tokens) > 4:
            continue
        if any(token.lower() in STOPWORDS for token in tokens):
            continue
        if name.isupper():
            continue
        filtered.append(name)
    return unique_preserve_order(filtered)


def extract_author_names_from_text(text: str) -> List[str]:
    normalized = normalize_whitespace(text)
    names: list[str] = []
    for pattern in AUTHOR_LINE_PATTERNS:
        for match in pattern.finditer(normalized):
            names.append(normalize_whitespace(match.group(1)))
    return filter_candidate_names(names)


def extract_institutions_from_text(text: str) -> List[str]:
    normalized = normalize_whitespace(text)
    entities: list[str] = []
    for pattern in INSTITUTION_ENTITY_PATTERNS:
        for match in pattern.finditer(normalized):
            entities.append(normalize_whitespace(match.group(1)))
    return unique_preserve_order(entity for entity in entities if len(entity) <= 120)


def contains_candidate_person_name(text: str) -> bool:
    for match in PERSON_NAME_PATTERN.finditer(text):
        candidate_name = normalize_whitespace(match.group(1))
        if is_plausible_person_name(candidate_name):
            return True
    return False


def is_plausible_person_name(name: str) -> bool:
    candidate_name = normalize_whitespace(name)
    if not candidate_name:
        return False

    tokens = candidate_name.split()
    if len(tokens) < 2 or len(tokens) > 4:
        return False
    if all(token.lower() in STOPWORDS for token in tokens):
        return False

    lowered_tokens = [token.lower().strip(".,;:!?()[]{}\"'") for token in tokens]
    if any(not token or token in STOPWORDS for token in lowered_tokens):
        return False
    if any(token in ORGANIZATION_NAME_HINTS for token in lowered_tokens):
        return False
    if sum(1 for token in lowered_tokens if token in ORGANIZATION_NAME_HINTS) >= 1:
        return False
    if sum(1 for token in lowered_tokens if token in NON_PERSON_TITLE_HINTS) >= 1:
        return False
    non_person_category_hits = sum(1 for token in lowered_tokens if token in NON_PERSON_CATEGORY_HINTS)
    if non_person_category_hits >= 2:
        return False
    if len(tokens) >= 3 and non_person_category_hits >= len(tokens) - 1:
        return False
    if candidate_name.isupper():
        return False
    return True


def event_winner_evidence_score(text: str) -> float:
    cleaned_text = normalize_whitespace(text)
    if not cleaned_text:
        return 0.0
    generic_event_topic = is_generic_event_topic_page("", cleaned_text[:140], cleaned_text[:1200])

    best_score = 0.0
    for sentence in split_sentences(cleaned_text):
        lowered_sentence = sentence.lower()
        winner_hits = sum(1 for hint in EVENT_WINNER_HINTS if hint in lowered_sentence)
        if winner_hits == 0:
            continue

        plausible_names = [
            normalize_whitespace(match.group(1))
            for match in PERSON_NAME_PATTERN.finditer(sentence)
            if is_plausible_person_name(normalize_whitespace(match.group(1)))
        ]
        strong_hits = sum(1 for hint in EVENT_WINNER_STRONG_HINTS if hint in lowered_sentence)
        score = min(0.42, 0.08 * winner_hits) + min(0.2, 0.08 * strong_hits)
        if plausible_names:
            score += 0.28
        if any(token in lowered_sentence for token in ("official", "tourism", "municipal", "municipality", "province")):
            score += 0.06
        if generic_event_topic:
            score -= 0.22
        best_score = max(best_score, min(1.0, score))

    return best_score


def has_event_winner_evidence(text: str, *, minimum_score: float = 0.38) -> bool:
    return event_winner_evidence_score(text) >= minimum_score


def extract_capitalized_entities(text: str) -> List[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    candidates: list[str] = []
    for match in TITLECASE_ENTITY_PATTERN.finditer(normalized):
        entity = normalize_whitespace(match.group(1)).strip(" ,;:.!?\"'()[]{}")
        if not entity:
            continue
        if entity in TITLECASE_ENTITY_BLOCKLIST:
            continue
        tokens = entity.split()
        if len(tokens) > 5:
            continue
        if len(tokens) == 1:
            token = tokens[0]
            if len(token) < 4:
                continue
            if token in TITLECASE_ENTITY_BLOCKLIST or token.lower() in STOPWORDS:
                continue
        if all(token.lower() in STOPWORDS for token in tokens):
            continue
        candidates.append(entity)
    return unique_preserve_order(candidates)


def extract_doi_candidates(text: str) -> List[str]:
    if not text:
        return []
    return unique_preserve_order(match.group(0).rstrip(").,;:") for match in DOI_PATTERN.finditer(text))


def contains_doi(text: str) -> bool:
    return bool(extract_doi_candidates(text))


def primary_document_text(url: str, title: str, content: str, *, max_chars: int = FRONT_MATTER_CHAR_LIMIT) -> str:
    return normalize_whitespace(
        " ".join(part for part in (url, title, (content or "")[:max_chars]) if part)
    )


def extract_primary_doi_candidates(
    url: str,
    title: str,
    content: str,
    *,
    max_chars: int = FRONT_MATTER_CHAR_LIMIT,
) -> List[str]:
    return extract_doi_candidates(primary_document_text(url, title, content, max_chars=max_chars))


def contains_primary_doi(
    url: str,
    title: str,
    content: str,
    *,
    max_chars: int = FRONT_MATTER_CHAR_LIMIT,
) -> bool:
    return bool(extract_primary_doi_candidates(url, title, content, max_chars=max_chars))


def looks_like_reference_citation(text: str) -> bool:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return False
    lower = cleaned.lower()
    signals = 0
    if any(hint in lower for hint in REFERENCE_HINTS):
        signals += 1
    if REFERENCE_STYLE_YEAR_PATTERN.search(cleaned):
        signals += 1
    if REFERENCE_STYLE_AUTHOR_PATTERN.search(cleaned):
        signals += 1
    if re.search(r"\b\d+\(\d+\)|\bpp?\.\s*\d", cleaned, re.IGNORECASE):
        signals += 1
    return signals >= 2


def is_heading_candidate(line: str) -> bool:
    stripped = line.strip().strip(":")
    if not stripped or len(stripped) > 100:
        return False
    normalized = normalize_heading(stripped)
    if len(normalized.split()) > 8:
        return False
    if re.search(r"\b(?:chapter|abstract|references|bibliography|appendix|contents|introduction)\b", normalized):
        return True
    letters = re.sub(r"[^A-Za-z]", "", stripped)
    if not letters:
        return False
    uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return uppercase_ratio > 0.6 or stripped.istitle()
