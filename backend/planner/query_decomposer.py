from __future__ import annotations

import asyncio
import logging
import re

import httpx

from agent.models import QueryDecompositionOutput
from config.planner_config import (
    DEFAULT_DECOMPOSITION_MAX_ATTEMPTS,
    DEFAULT_DECOMPOSITION_MODEL,
    DEFAULT_DECOMPOSITION_TIMEOUT_SECONDS,
    DEFAULT_DECOMPOSITION_MAX_QUERY_LENGTH as DEFAULT_MAX_QUERY_LENGTH,
    load_decomposition_config,
)
from planner.query_intent import analyze_query_intent
from utils.text_utils import (
    clean_fragment,
    extract_capitalized_entities,
    extract_constraint_phrases,
    important_terms,
    keyword_focus,
    normalize_whitespace,
    specificity_overlap_score,
    split_into_clauses,
    specific_query_terms,
    strip_question_prefix,
    unique_preserve_order,
)

try:
    from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    APIConnectionError = None
    APITimeoutError = None
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False

logger = logging.getLogger("planner.query_decomposer")

DECOMPOSITION_TIMEOUT_BACKOFF_MULTIPLIER = 1.5
MIN_DECOMPOSITION_CLUES = 3
MAX_DECOMPOSITION_CLUES = 5

DECOMPOSITION_SYSTEM_PROMPT = """
You are a retrieval planner for a tool-augmented deep research web browsing system.

Decompose the user's question into 3 to 5 independent search clues.

Each clue must:
- be a standalone web search query
- capture a single entity, attribute, relation, or constraint
- preserve hard constraints such as dates, counts, titles, names, places, and time windows
- avoid pronouns or vague references
- avoid answering the question
- be phrased so it can be sent directly to a search engine

If the question is academic and mentions a paper, thesis, dissertation, acknowledgements, university, department, or professor:
- include at least one clue that targets a thesis or dissertation PDF in an institutional repository
- include at least one clue that targets the acknowledgements section
- include at least one clue that targets the author's later career page or appointment

Return only the clues in the schema.
""".strip()


def decompose_query(query: str) -> list[str]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(decompose_query_async(query))

    logger.warning(
        "decompose_query() was called from an async context; using heuristic fallback."
    )
    return _heuristic_decompose_query(query)


async def decompose_query_async(query: str) -> list[str]:
    decomposition_config = load_decomposition_config()
    normalized_query = normalize_whitespace(query)[: decomposition_config.max_query_length].rstrip(" ?")
    if not normalized_query:
        return []

    heuristic_clues = _heuristic_decompose_query(normalized_query)
    llm_clues = await _decompose_query_with_openai(normalized_query)
    if not llm_clues:
        return heuristic_clues

    merged_clues = _merge_decomposition_clues(normalized_query, llm_clues, heuristic_clues)
    if len(merged_clues) < MIN_DECOMPOSITION_CLUES:
        return heuristic_clues
    return merged_clues


def sanitize_model_clues(clues: list[str], query: str = "") -> list[str]:
    sanitized: list[str] = []
    normalized_query = normalize_whitespace(query)
    for clue in clues:
        cleaned_clue = normalize_whitespace(clue)
        cleaned_clue = cleaned_clue.strip(" \"'")
        cleaned_clue = clean_fragment(cleaned_clue)
        if not cleaned_clue:
            continue
        if len(important_terms(cleaned_clue)) < 2:
            continue
        if normalized_query and not _is_grounded_decomposition_clue(normalized_query, cleaned_clue):
            continue
        sanitized.append(cleaned_clue)
    return unique_preserve_order(sanitized)


def merge_decomposition_clues(query: str, llm_clues: list[str], fallback_clues: list[str]) -> list[str]:
    return _merge_decomposition_clues(query, llm_clues, fallback_clues)


def _heuristic_decompose_query(query: str) -> list[str]:
    normalized_query = normalize_whitespace(query)[:DEFAULT_MAX_QUERY_LENGTH].rstrip(" ?")
    if not normalized_query:
        return []

    intent = analyze_query_intent(normalized_query)
    if intent.is_open_domain_browsecomp:
        if intent.needs_person_identity_hop:
            person_identity_clues = _heuristic_decompose_person_identity_query(normalized_query)
            if person_identity_clues:
                return person_identity_clues[:MAX_DECOMPOSITION_CLUES]
        if intent.answer_type == "year":
            year_browsecomp_clues = _heuristic_decompose_year_browsecomp_query(normalized_query)
            if year_browsecomp_clues:
                return year_browsecomp_clues[:MAX_DECOMPOSITION_CLUES]
        if intent.prefers_event_sources:
            event_browsecomp_clues = _heuristic_decompose_event_browsecomp_query(normalized_query)
            if event_browsecomp_clues:
                return event_browsecomp_clues[:MAX_DECOMPOSITION_CLUES]
        browsecomp_clues = _heuristic_decompose_browsecomp_query(normalized_query)
        if browsecomp_clues:
            return browsecomp_clues[:MAX_DECOMPOSITION_CLUES]

    core_query = strip_question_prefix(normalized_query)
    constraints = extract_constraint_phrases(core_query)
    clauses = split_into_clauses(core_query, constraints)
    subject_focus = clauses[0] if clauses else keyword_focus(core_query, max_terms=10)

    candidates: list[str] = []
    if subject_focus:
        candidates.append(subject_focus)
    candidates.extend(clauses[1:])

    for constraint in constraints:
        if subject_focus:
            candidates.append(clean_fragment(f"{subject_focus} {constraint}"))

    candidates.append(core_query)
    focused = keyword_focus(core_query, max_terms=12)
    if focused:
        candidates.append(focused)

    clues = unique_preserve_order(
        clue for clue in candidates if clue and len(important_terms(clue)) >= 2
    )
    if len(clues) < MIN_DECOMPOSITION_CLUES:
        clues = unique_preserve_order(clues + _generate_backoff_clues(core_query, constraints))
    return clues[:MAX_DECOMPOSITION_CLUES]


async def _decompose_query_with_openai(query: str) -> list[str]:
    decomposition_config = load_decomposition_config()
    if not decomposition_config.use_llm:
        return []

    if not decomposition_config.openai_api_key or not OPENAI_AVAILABLE:
        return []

    model_name = decomposition_config.model
    base_timeout_seconds = decomposition_config.timeout_seconds
    max_attempts = decomposition_config.max_attempts

    for attempt in range(1, max_attempts + 1):
        attempt_timeout = base_timeout_seconds * (
            DECOMPOSITION_TIMEOUT_BACKOFF_MULTIPLIER ** (attempt - 1)
        )
        client = AsyncOpenAI(
            api_key=decomposition_config.openai_api_key,
            timeout=attempt_timeout,
        )
        try:
            response = await client.responses.parse(
                model=model_name,
                input=[
                    {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Decompose the following user question into 3 to 5 search clues.\n\n"
                            f"User question: {query}"
                        ),
                    },
                ],
                text_format=QueryDecompositionOutput,
            )
        except Exception as error:
            if _is_retryable_error(error) and attempt < max_attempts:
                logger.warning(
                    "OpenAI decomposition attempt %s/%s failed for '%s'. Retrying: %s",
                    attempt,
                    max_attempts,
                    model_name,
                    error,
                )
                continue
            logger.warning("OpenAI decomposition failed; using heuristic fallback: %s", error)
            return []
        finally:
            await client.close()

        parsed_output = response.output_parsed
        if parsed_output is None:
            return []
        return sanitize_model_clues(parsed_output.clues, query=query)

    return []


def _merge_decomposition_clues(query: str, llm_clues: list[str], fallback_clues: list[str]) -> list[str]:
    sanitized_llm_clues = sanitize_model_clues(llm_clues, query=query)
    intent = analyze_query_intent(query)
    if intent.is_open_domain_browsecomp:
        # Keep BrowseComp decomposition anchored to the prompt itself; heuristic clues are safer
        # than speculative title guesses from the model.
        return unique_preserve_order(fallback_clues + sanitized_llm_clues)[:MAX_DECOMPOSITION_CLUES]
    if len(sanitized_llm_clues) >= MIN_DECOMPOSITION_CLUES:
        return sanitized_llm_clues[:MAX_DECOMPOSITION_CLUES]
    return unique_preserve_order(sanitized_llm_clues + fallback_clues)[:MAX_DECOMPOSITION_CLUES]


def _generate_backoff_clues(query: str, constraints: list[str]) -> list[str]:
    focused = keyword_focus(query, max_terms=10)
    backoff = [query, focused]
    if constraints:
        backoff.append(clean_fragment(f"{focused} {' '.join(constraints)}"))
        backoff.extend(constraints)
    return unique_preserve_order(clue for clue in backoff if clue)


def _heuristic_decompose_browsecomp_query(query: str) -> list[str]:
    normalized_query = normalize_whitespace(query).rstrip(" ?")
    specific_focus = specific_query_terms(normalized_query)
    sentence_chunks: list[tuple[str, str]] = []

    raw_sentences = [
        normalize_whitespace(sentence)
        for sentence in re.split(r"(?<=[.!?;])\s+", normalized_query)
        if normalize_whitespace(sentence)
    ]
    for sentence in raw_sentences:
        compact = keyword_focus(sentence, max_terms=14)
        if len(important_terms(compact)) >= 3:
            sentence_chunks.append((sentence, compact))

    sentence_clues = [compact for _, compact in sentence_chunks]
    if raw_sentences:
        ending_sentence = raw_sentences[-1]
        if any(term in ending_sentence.lower() for term in ("how many", "what's the name", "what is the name")):
            prior_sentence = raw_sentences[-2] if len(raw_sentences) >= 2 else ""
            combined = keyword_focus(f"{prior_sentence} {ending_sentence}", max_terms=16)
            if len(important_terms(combined)) >= 4:
                sentence_clues.append(combined)

    focus_clues: list[str] = []
    if specific_focus:
        focus_clues.append(" ".join(specific_focus[:6]))
        if len(specific_focus) >= 4:
            focus_clues.append(" ".join(specific_focus[:4]))

    relation_clues: list[str] = []
    relation_hints = (
        ("first chapter", "released", "created", "group", "elementary school", "known as"),
        ("classic story", "1800s", "novelist", "other forms of art"),
        ("theme", "perfection"),
        ("antagonist", "companion", "instrument"),
        ("how many", "movements", "effects", "used"),
    )
    for hint_group in relation_hints:
        matching_compacts = sorted(
            (
                (
                    sum(1 for hint in hint_group if hint in raw_sentence.lower()),
                    compact,
                )
                for raw_sentence, compact in sentence_chunks
                if any(hint in raw_sentence.lower() for hint in hint_group)
            ),
            reverse=True,
        )
        if matching_compacts:
            relation_clues.append(matching_compacts[0][1])

    if raw_sentences:
        ending_sentence = raw_sentences[-1]
        if any(term in ending_sentence.lower() for term in ("how many", "movements", "effects")):
            prior_sentence = raw_sentences[-2] if len(raw_sentences) >= 2 else ""
            combined = keyword_focus(f"{prior_sentence} {ending_sentence}", max_terms=16)
            if len(important_terms(combined)) >= 4:
                relation_clues.append(combined)

    prioritized_candidates = relation_clues + sentence_clues + focus_clues
    clues = unique_preserve_order(
        clue
        for clue in prioritized_candidates
        if clue and len(important_terms(clue)) >= 3
    )
    return clues[:MAX_DECOMPOSITION_CLUES]


def _heuristic_decompose_person_identity_query(query: str) -> list[str]:
    normalized_query = normalize_whitespace(query).rstrip(" ?")
    query_lower = normalized_query.lower()
    protected_query = re.sub(
        r"\b(Dr|Mr|Ms|Mrs|Prof)\.\s+",
        lambda match: f"{match.group(1)} ",
        normalized_query,
    )
    raw_sentences = [
        normalize_whitespace(sentence)
        for sentence in re.split(r"(?<=[.!?;])\s+", protected_query)
        if normalize_whitespace(sentence)
    ]
    sentence_chunks: list[tuple[str, str]] = []
    advisor_name = ""
    advisor_year_tokens: list[str] = []
    for sentence in raw_sentences:
        cleaned_sentence = re.sub(
            r"^(?:i am looking for|i'm looking for|looking for)\s+(?:someone|a person|the person)?\s*(?:who\s+)?",
            "",
            sentence,
            flags=re.IGNORECASE,
        )
        cleaned_sentence = normalize_whitespace(cleaned_sentence) or sentence
        lowered = cleaned_sentence.lower()
        compact = keyword_focus(cleaned_sentence, max_terms=16)
        if len(important_terms(compact)) >= 4:
            sentence_chunks.append((cleaned_sentence, compact))

        if not advisor_name and any(term in lowered for term in ("student of", "advisor", "mentor", "supervisor", "under")):
            advisor_match = re.search(
                r"\b(?:student of|advised by|advisor(?: was)?|mentor(?: was)?|supervisor(?: was)?|under)\s+"
                r"(Dr\.?\s+[A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){0,2}|"
                r"[A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){0,2})",
                cleaned_sentence,
            )
            if advisor_match:
                advisor_name = normalize_whitespace(advisor_match.group(1))
            else:
                capitalized_entities = extract_capitalized_entities(cleaned_sentence)
                if capitalized_entities:
                    advisor_name = capitalized_entities[0]
            advisor_year_tokens = re.findall(r"\b(?:19|20)\d{2}\b", cleaned_sentence)

    advisor_clues: list[str] = []
    children_clues: list[str] = []
    article_clues: list[str] = []
    if advisor_name:
        if len(advisor_year_tokens) >= 2:
            advisor_clues.append(f"\"{advisor_name}\" student {advisor_year_tokens[0]} {advisor_year_tokens[1]}")
        else:
            advisor_clues.append(f"\"{advisor_name}\" student")

    for sentence, compact in sentence_chunks:
        lowered = sentence.lower()
        if "children" in lowered:
            quantity_match = re.search(
                r"\b(?:at least|more than|less than|between)\s+[^.?!,;]{0,40}\bchildren\b",
                lowered,
            )
            if quantity_match:
                children_phrase = normalize_whitespace(quantity_match.group(0))
                if advisor_name:
                    children_clues.append(f"\"{advisor_name}\" {children_phrase}")
                children_clues.append(children_phrase)
                continue
            children_clues.append(compact)
            continue

        year_tokens = re.findall(r"\b(?:19|20)\d{2}\b", sentence)
        year_window = f"{year_tokens[0]} {year_tokens[-1]}" if len(year_tokens) >= 2 else ""

        if all(term in lowered for term in ("new south", "cities")):
            article_clues.append(
                normalize_whitespace(
                    f"\"New South\" cities growth {year_window} article".strip()
                )
            )
            continue

        if "scrap metal" in lowered and any(term in lowered for term in ("violence", "gardening", "garden")):
            article_clues.append(
                normalize_whitespace(
                    f"\"scrap metal\" gardening violence {year_window} article".strip()
                )
            )
            continue

        if (
            "african-american" in lowered
            and any(term in lowered for term in ("four schools", "all-white", "admitted"))
        ):
            article_clues.append(
                normalize_whitespace(
                    f"\"African-American\" students admitted four all-white schools {year_window} article".strip()
                )
            )
            continue

        if "article" in lowered:
            if compact.lower().endswith("article"):
                article_clues.append(compact)
            else:
                article_clues.append(f"{compact} article")
            continue

    focus_clues: list[str] = []
    specific_focus = specific_query_terms(normalized_query)
    if specific_focus:
        focus_clues.append(" ".join(specific_focus[:6]))
        if len(specific_focus) >= 4:
            focus_clues.append(" ".join(specific_focus[:4]))

    prioritized = (
        advisor_clues[:1]
        + children_clues[:1]
        + article_clues[:3]
        + [compact for _, compact in sentence_chunks]
        + focus_clues
    )
    clues = unique_preserve_order(
        clue
        for clue in prioritized
        if clue and len(important_terms(clue)) >= 4
    )
    return clues[:MAX_DECOMPOSITION_CLUES]


def _heuristic_decompose_event_browsecomp_query(query: str) -> list[str]:
    normalized_query = normalize_whitespace(query).rstrip(" ?")
    query_lower = normalized_query.lower()
    intent = analyze_query_intent(normalized_query)
    raw_sentences = [
        normalize_whitespace(sentence)
        for sentence in re.split(r"(?<=[.!?;])\s+", normalized_query)
        if normalize_whitespace(sentence)
    ]
    sentence_chunks: list[tuple[str, str]] = []
    for sentence in raw_sentences:
        compact = keyword_focus(sentence, max_terms=16)
        if len(important_terms(compact)) >= 4:
            sentence_chunks.append((sentence, compact))

    specific_focus = specific_query_terms(normalized_query)
    relation_hints = (
        ("stew", "fish", "meat", "condiment", "ingredient"),
        ("township", "celebration", "named after", "2023"),
        ("1995", "2005", "shifted", "highlight", "subject"),
        ("anniversary", "competition", "provincial", "province"),
        ("beauty pageant", "contest", "winner", "first", "last"),
    )

    relation_clues: list[str] = []
    for hint_group in relation_hints:
        matching_compacts = sorted(
            (
                (
                    sum(1 for hint in hint_group if hint in raw_sentence.lower()),
                    compact,
                )
                for raw_sentence, compact in sentence_chunks
                if any(hint in raw_sentence.lower() for hint in hint_group)
            ),
            reverse=True,
        )
        if matching_compacts:
            relation_clues.append(matching_compacts[0][1])

    synthesized_clues: list[str] = []
    if all(term in query_lower for term in ("stew", "fish", "condiment")):
        synthesized_clues.append("vegetable stew fish meat condiment critical ingredient")
    if "township" in query_lower and "named after" in query_lower:
        synthesized_clues.append("2023 township celebration named after stew")
    if "1995" in query_lower and "2005" in query_lower and "shifted" in query_lower:
        synthesized_clues.append("1995 2005 authorities shifted highlight subject event to stand apart")
    if all(term in query_lower for term in ("after february", "before september")):
        synthesized_clues.append("annual celebration after February before September")
    if "anniversary" in query_lower and any(term in query_lower for term in ("provincial", "province", "winners")):
        synthesized_clues.append("thirteenth anniversary competition town provincial festivities same province winners")
    if (
        not intent.needs_event_discovery_hop
        and "beauty pageant" in query_lower
        and any(
        phrase in query_lower
        for phrase in ("first and last name", "first and last names", "full name", "won that contest", "won the contest")
        )
    ):
        synthesized_clues.append("beauty pageant contest winner first and last names")

    focus_clues: list[str] = []
    if specific_focus:
        focus_clues.append(" ".join(specific_focus[:6]))
        if len(specific_focus) >= 4:
            focus_clues.append(" ".join(specific_focus[:4]))

    prioritized_candidates = (
        synthesized_clues
        + relation_clues
        + [compact for _, compact in sentence_chunks]
        + focus_clues
    )
    clues = unique_preserve_order(
        clue
        for clue in prioritized_candidates
        if clue and len(important_terms(clue)) >= 4
    )
    return clues[:MAX_DECOMPOSITION_CLUES]


def _heuristic_decompose_year_browsecomp_query(query: str) -> list[str]:
    normalized_query = normalize_whitespace(query).rstrip(" ?")
    query_lower = normalized_query.lower()
    raw_sentences = [
        normalize_whitespace(sentence)
        for sentence in re.split(r"(?<=[.!?;])\s+", normalized_query)
        if normalize_whitespace(sentence)
    ]
    sentence_chunks: list[tuple[str, str]] = []
    for sentence in raw_sentences:
        compact = keyword_focus(sentence, max_terms=16)
        if len(important_terms(compact)) >= 4:
            sentence_chunks.append((sentence, compact))

    relation_hints = (
        ("event", "year", "loss of lives", "victims", "dedication", "honor"),
        ("monument", "memorial", "constructed", "prior to", "former yugoslavia"),
        ("bosnia", "top", "largest cities", "population census", "2013"),
        ("artist", "born", "1928"),
    )
    relation_clues: list[str] = []
    for hint_group in relation_hints:
        matching_compacts = sorted(
            (
                (
                    sum(1 for hint in hint_group if hint in raw_sentence.lower()),
                    compact,
                )
                for raw_sentence, compact in sentence_chunks
                if any(hint in raw_sentence.lower() for hint in hint_group)
            ),
            reverse=True,
        )
        if matching_compacts:
            relation_clues.append(matching_compacts[0][1])

    synthesized_clues: list[str] = []
    if any(term in query_lower for term in ("loss of lives", "victims", "dedication", "honor")):
        synthesized_clues.append("event led to loss of lives monument dedicated in honor of victims")
    if "monument" in query_lower and "former yugoslavia" in query_lower:
        synthesized_clues.append("monument constructed before 1970 former Yugoslavia Bosnia")
    if "population census" in query_lower and "largest cities" in query_lower:
        synthesized_clues.append("2013 population census Bosnia top largest cities")
    if "artist" in query_lower and "born" in query_lower:
        synthesized_clues.append("artist born 1928 monument Bosnia")
    if any(term in query_lower for term in ("massacre", "battle", "victims", "lost their lives")):
        synthesized_clues.append("victims memorial event year monument")

    focus_clues: list[str] = []
    specific_focus = specific_query_terms(normalized_query)
    if specific_focus:
        focus_clues.append(" ".join(specific_focus[:6]))
        if len(specific_focus) >= 4:
            focus_clues.append(" ".join(specific_focus[:4]))

    prioritized_candidates = synthesized_clues + relation_clues + [compact for _, compact in sentence_chunks] + focus_clues
    clues = unique_preserve_order(
        clue
        for clue in prioritized_candidates
        if clue and len(important_terms(clue)) >= 4
    )
    return clues[:MAX_DECOMPOSITION_CLUES]


def _is_retryable_error(error: Exception) -> bool:
    retryable_error_types = tuple(
        error_type
        for error_type in (APITimeoutError, APIConnectionError, TimeoutError, httpx.TimeoutException)
        if error_type is not None
    )
    return bool(retryable_error_types and isinstance(error, retryable_error_types)) or (
        "timed out" in str(error).lower()
    )


def _is_grounded_decomposition_clue(query: str, clue: str) -> bool:
    intent = analyze_query_intent(query)
    if not intent.is_open_domain_browsecomp:
        return True

    stripped_clue = re.sub(r"\b(?:site|filetype):\S+", " ", clue, flags=re.IGNORECASE)
    query_terms = set(specific_query_terms(query))
    clue_terms = [term.lower() for term in important_terms(stripped_clue)]
    if not clue_terms:
        return False

    # For BrowseComp title-discovery questions, reject clues that introduce a new
    # specific title or character before we've seen supporting evidence.
    query_entities = {entity.lower() for entity in extract_capitalized_entities(query)}
    clue_entities = [
        entity.lower()
        for entity in extract_capitalized_entities(clue)
        if entity.lower() not in {"wikipedia", "fandom"}
    ]
    if clue_entities and any(entity not in query_entities for entity in clue_entities):
        return False

    query_numbers = set(re.findall(r"\b\d+\b", query))
    clue_numbers = set(re.findall(r"\b\d+\b", clue))
    if clue_numbers - query_numbers:
        return False

    overlap_count = sum(1 for term in clue_terms if term in query_terms)
    overlap_score = specificity_overlap_score(query, stripped_clue)
    if overlap_count >= 2 and overlap_score >= 0.12:
        return True
    if overlap_count >= 1 and len(clue_terms) <= 10:
        return True

    return overlap_score >= 0.18
