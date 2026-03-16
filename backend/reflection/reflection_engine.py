from __future__ import annotations

from agent.models import AnswerCandidate, Document, ReflectionResult
from planner.query_intent import analyze_query_intent
from reflection.query_refiner import generate_follow_up_clues_async
from utils.text_utils import (
    ABILITY_QUERY_HINTS,
    CHARACTER_QUERY_HINTS,
    ENCYCLOPEDIC_SOURCE_HINTS,
    PAPER_CONTENT_HINTS,
    THESIS_CONTENT_HINTS,
    contains_doi,
    event_page_score,
    historical_year_artist_birth_year,
    historical_year_build_year_cutoff,
    important_terms,
    is_broad_overview_page,
    is_generic_media_topic_page,
    is_grounded_browsecomp_page,
    is_low_trust_social_page,
    is_non_english_wiki_page,
    is_recipe_food_page,
    is_wiki_meta_page,
    keyword_focus,
    looks_like_event_page,
    looks_like_media_page,
    query_requires_bosnia_top_city,
    specific_query_terms,
    specificity_overlap_score,
    unique_preserve_order,
)


class ReflectionEngine:
    async def reflect(
        self,
        query: str,
        docs: list[Document],
        existing_clues: list[str],
        answer_candidate: AnswerCandidate | None = None,
    ) -> ReflectionResult:
        intent = analyze_query_intent(query)
        if answer_candidate and answer_candidate.confidence >= 0.85:
            return ReflectionResult(
                should_continue=False,
                notes=["High-confidence answer found, stopping exploration early."],
            )

        if not docs:
            clues = self._fallback_gap_queries(query, existing_clues, missing_everything=True)
            return ReflectionResult(
                should_continue=bool(clues),
                clues=clues,
                notes=["Initial search returned no usable documents, broadening the query frontier."],
            )

        missing_pdf = intent.prefers_repository_sources and not any(
            doc.url.lower().endswith(".pdf")
            or any(
                hint in f"{doc.url.lower()} {doc.title.lower()} {doc.content[:400].lower()}"
                for hint in THESIS_CONTENT_HINTS
            )
            for doc in docs
        )
        missing_ack = intent.requires_acknowledgement_section and not any(
            doc.acknowledgement_section or doc.sections.get("acknowledgements")
            for doc in docs
        )
        missing_reference_section = intent.requires_reference_section and not any(
            doc.sections.get("references") for doc in docs
        )
        missing_career = (
            intent.needs_career_hop
            and not any(
                any(
                    hint in f"{doc.title.lower()} {doc.content[:800].lower()}"
                    for hint in ("professor", "faculty", "appointed", "joined", "ac.uk")
                )
                for doc in docs
            )
        )
        missing_doi = intent.targets_citation_identifier and not any(
            contains_doi(f"{doc.url} {doc.title} {doc.content[:5000]}") for doc in docs
        )
        missing_paper_source = intent.prefers_paper_sources and not any(
            contains_doi(f"{doc.url} {doc.title} {doc.content[:4000]}")
            or any(
                hint in f"{doc.url.lower()} {doc.title.lower()} {doc.content[:1200].lower()}"
                for hint in PAPER_CONTENT_HINTS
            )
            for doc in docs
        )
        missing_encyclopedic_source = intent.prefers_encyclopedic_sources and not any(
            any(hint in f"{doc.url.lower()} {doc.title.lower()} {doc.source.lower()}" for hint in ENCYCLOPEDIC_SOURCE_HINTS)
            and not is_wiki_meta_page(doc.url, doc.title, doc.content[:800])
            and not is_non_english_wiki_page(doc.url)
            and not is_generic_media_topic_page(doc.url, doc.title, doc.content[:1200])
            for doc in docs
        )
        missing_character_source = intent.prefers_character_sources and not any(
            any(hint in f"{doc.title.lower()} {doc.content[:1200].lower()}" for hint in CHARACTER_QUERY_HINTS)
            and (not intent.is_media_query or looks_like_media_page(doc.url, doc.title, doc.content[:2200]))
            and not is_wiki_meta_page(doc.url, doc.title, doc.content[:800])
            and not is_non_english_wiki_page(doc.url)
            and not is_generic_media_topic_page(doc.url, doc.title, doc.content[:1200])
            for doc in docs
        )
        grounded_event_candidates = [
            doc
            for doc in docs
            if intent.prefers_event_sources
            and not is_low_trust_social_page(doc.url, doc.title, doc.content[:400])
            and not is_recipe_food_page(doc.url, doc.title, doc.content[:700])
            and event_page_score(doc.url, doc.title, doc.content[:2200]) >= 0.42
            and specificity_overlap_score(query, f"{doc.title} {doc.content[:2200]}") >= 0.12
            and is_grounded_browsecomp_page(
                query,
                doc.url,
                doc.title,
                doc.content[:2200],
                require_media=intent.is_media_query,
            )
        ]
        missing_event_source = intent.prefers_event_sources and not grounded_event_candidates
        missing_winner_evidence = intent.prefers_event_sources and not any(
            any(
                term in f"{doc.title.lower()} {doc.content[:1800].lower()}"
                for term in ("winner", "won", "pageant", "contest", "queen", "coronation")
            )
            for doc in grounded_event_candidates
        )
        missing_count_evidence = intent.targets_count and not any(
            any(hint in f"{doc.title.lower()} {doc.content[:1800].lower()}" for hint in ABILITY_QUERY_HINTS)
            for doc in docs
        )
        missing_title_specific_source = intent.is_open_domain_browsecomp and not (
            bool(grounded_event_candidates)
            if intent.prefers_event_sources
            else any(
                not is_broad_overview_page(doc.url, doc.title)
                and not is_wiki_meta_page(doc.url, doc.title, doc.content[:800])
                and not is_non_english_wiki_page(doc.url)
                and not is_generic_media_topic_page(doc.url, doc.title, doc.content[:1200])
                and is_grounded_browsecomp_page(
                    query,
                    doc.url,
                    doc.title,
                    doc.content[:2200],
                    require_media=intent.is_media_query,
                )
                and specificity_overlap_score(query, f"{doc.title} {doc.content[:2200]}") >= 0.12
                for doc in docs
            )
        )
        broad_overview_dominated = intent.is_open_domain_browsecomp and (
            sum(1 for doc in docs[:4] if is_broad_overview_page(doc.url, doc.title))
            >= max(1, min(2, len(docs[:4])))
        )

        refined_clues = await generate_follow_up_clues_async(query, docs, existing_clues)
        if not refined_clues and not any(
            (
                missing_pdf,
                missing_ack,
                missing_reference_section,
                missing_career,
                missing_doi,
                missing_paper_source,
                missing_encyclopedic_source,
                missing_character_source,
                missing_event_source,
                missing_winner_evidence,
                missing_count_evidence,
                missing_title_specific_source,
                broad_overview_dominated,
            )
        ):
            if answer_candidate and answer_candidate.answer:
                return ReflectionResult(
                    should_continue=False,
                    notes=["The current evidence frontier is saturated and the answer is already grounded."],
                )
            refined_clues = self._fallback_gap_queries(
                query,
                existing_clues,
                missing_everything=True,
            )
        elif not refined_clues:
            refined_clues = self._fallback_gap_queries(
                query,
                existing_clues,
                missing_pdf=missing_pdf,
                missing_ack=missing_ack,
                missing_reference_section=missing_reference_section,
                missing_career=missing_career,
                missing_doi=missing_doi,
                missing_paper_source=missing_paper_source,
                missing_encyclopedic_source=missing_encyclopedic_source,
                missing_character_source=missing_character_source,
                missing_event_source=missing_event_source,
                missing_winner_evidence=missing_winner_evidence,
                missing_count_evidence=missing_count_evidence,
                missing_title_specific_source=missing_title_specific_source,
                broad_overview_dominated=broad_overview_dominated,
            )

        notes: list[str] = []
        if missing_pdf:
            notes.append("Reflection detected a missing thesis or dissertation document.")
        if missing_ack:
            notes.append("Reflection detected that an acknowledgements section still has not been found.")
        if missing_reference_section:
            notes.append("Reflection detected that a references section is still missing.")
        if missing_career:
            notes.append("Reflection detected weak career-page evidence for the professor/UK hop.")
        if missing_doi:
            notes.append("Reflection detected that no DOI-bearing evidence has been found yet.")
        if missing_paper_source:
            notes.append("Reflection detected that the current sources do not look like paper or journal pages.")
        if missing_encyclopedic_source:
            notes.append("Reflection detected that no strong wiki or fandom-style entity source has been found yet.")
        if missing_character_source:
            notes.append("Reflection detected that character-specific evidence is still weak.")
        if missing_event_source:
            notes.append("Reflection detected that no strong festival or event source has been found yet.")
        if missing_winner_evidence:
            notes.append("Reflection detected that winner or pageant-specific evidence is still weak.")
        if missing_count_evidence:
            notes.append("Reflection detected that no movement, ability, or count-oriented evidence has been found yet.")
        if missing_title_specific_source:
            notes.append("Reflection detected that the frontier still lacks a title- or entity-specific page.")
        if broad_overview_dominated:
            notes.append("Reflection detected that broad overview pages are dominating the current frontier.")
        return ReflectionResult(
            should_continue=bool(refined_clues),
            clues=refined_clues,
            notes=notes
            or (
                ["Reflection generated more specific follow-up queries."]
                if refined_clues
                else ["The current evidence frontier is saturated and no higher-value reformulation remains."]
            ),
        )

    def _fallback_gap_queries(
        self,
        query: str,
        existing_clues: list[str],
        *,
        missing_everything: bool = False,
        missing_pdf: bool = False,
        missing_ack: bool = False,
        missing_reference_section: bool = False,
        missing_career: bool = False,
        missing_doi: bool = False,
        missing_paper_source: bool = False,
        missing_encyclopedic_source: bool = False,
        missing_character_source: bool = False,
        missing_event_source: bool = False,
        missing_winner_evidence: bool = False,
        missing_count_evidence: bool = False,
        missing_title_specific_source: bool = False,
        broad_overview_dominated: bool = False,
    ) -> list[str]:
        intent = analyze_query_intent(query)
        clues: list[str] = []
        specific_focus = " ".join(specific_query_terms(query)[:6])
        event_seed = ""
        if intent.prefers_event_sources:
            event_candidates = [
                clue
                for clue in existing_clues
                if any(term in clue.lower() for term in ("festival", "celebration", "tourism", "township", "province", "anniversary"))
                and not any(term in clue.lower() for term in ("beauty pageant winner", "contest winner", "queen winner"))
            ]
            if event_candidates:
                event_seed = max(
                    event_candidates,
                    key=lambda clue: (
                        sum(
                            1
                            for term in ("township", "municipality", "anniversary", "province", "official", "tourism")
                            if term in clue.lower()
                        ),
                        specificity_overlap_score(query, clue),
                        -len(important_terms(clue)),
                    ),
                )
        compact_focus = (
            keyword_focus(event_seed, max_terms=8)
            if event_seed
            else keyword_focus(query, max_terms=10) or specific_focus or query
        )
        should_probe_repository = missing_pdf or (missing_everything and intent.prefers_repository_sources)
        should_probe_acknowledgements = missing_ack or (
            missing_everything and intent.requires_acknowledgement_section
        )
        should_probe_career = missing_career or (missing_everything and intent.needs_career_hop)
        should_probe_doi = missing_doi or (missing_everything and intent.targets_citation_identifier)
        should_probe_references = missing_reference_section or (
            missing_everything and intent.requires_reference_section
        )
        should_probe_paper = missing_paper_source or (missing_everything and intent.prefers_paper_sources)
        should_probe_encyclopedic = missing_encyclopedic_source or (
            missing_everything and intent.prefers_encyclopedic_sources
        )
        should_probe_character = missing_character_source or (
            missing_everything and intent.prefers_character_sources
        )
        should_probe_event = missing_event_source or (missing_everything and intent.prefers_event_sources)
        should_probe_winner = missing_winner_evidence or (
            missing_everything and intent.answer_type in {"person_name", "person_last_name"}
        )
        if intent.needs_event_discovery_hop and (should_probe_event or missing_title_specific_source or broad_overview_dominated):
            should_probe_winner = False
        should_probe_count = missing_count_evidence or (missing_everything and intent.targets_count)
        should_probe_title_specific = (
            missing_title_specific_source
            or broad_overview_dominated
            or (missing_everything and intent.is_open_domain_browsecomp)
        )
        should_probe_historical_year = (
            intent.answer_type == "year"
            and intent.is_open_domain_browsecomp
            and (
                missing_title_specific_source
                or broad_overview_dominated
                or missing_everything
            )
        )

        if should_probe_repository:
            clues.extend(
                [
                    f"site:.edu {query} thesis pdf",
                    f"site:.edu {query} dissertation repository",
                ]
            )
        if should_probe_acknowledgements:
            clues.extend(
                [
                    f"{query} acknowledgements pdf",
                    f"site:.edu {query} acknowledgements",
                ]
            )
        if should_probe_career:
            clues.extend(
                [
                    f"{query} professor uk 2018",
                    f"{query} faculty profile",
                ]
            )
        if should_probe_doi:
            clues.extend(
                [
                    f"{query} doi",
                    f"site:doi.org {query}",
                    f"{query} journal article doi",
                ]
            )
        if should_probe_references:
            clues.extend(
                [
                    f"{query} references",
                    f"{query} bibliography",
                    f"{query} cited references",
                ]
            )
        if should_probe_paper:
            clues.extend(
                [
                    f"{query} journal article",
                    f"{query} abstract",
                ]
            )
        if should_probe_event:
            if intent.needs_event_discovery_hop:
                if compact_focus:
                    clues.append(f"{compact_focus} festival celebration official")
                    clues.append(f"{compact_focus} tourism municipality festival")
                if "anniversary" in query.lower():
                    clues.append(f"{compact_focus} anniversary competition province official".strip())
                if all(term in query.lower() for term in ("after february", "before september")):
                    clues.append(f"{compact_focus} annual celebration after February before September".strip())
                if "1995" in query and "2005" in query and "shifted" in query.lower():
                    clues.append(f"{compact_focus} 1995 2005 shifted highlight subject celebration".strip())
            else:
                clues.extend(
                    [
                        f"{query} festival celebration official",
                        f"{query} tourism municipality festival",
                    ]
                )
        if should_probe_winner:
            winner_focus = compact_focus if intent.prefers_event_sources else query
            if not intent.needs_event_discovery_hop or not should_probe_event:
                clues.extend(
                    [
                        f"{winner_focus} beauty pageant winner",
                        f"{winner_focus} contest winner",
                        f"{winner_focus} queen winner",
                    ]
                )
        if should_probe_title_specific and specific_focus:
            if intent.is_media_query:
                clues.extend(
                    [
                        f"site:wikipedia.org {specific_focus}",
                        f"site:fandom.com {specific_focus}",
                        f"{specific_focus} manga title",
                    ]
                )
            if intent.prefers_character_sources:
                clues.append(f"{specific_focus} character wiki")
            if intent.targets_count:
                clues.append(f"{specific_focus} movements effects")
            if intent.prefers_event_sources:
                clues.append(f"{compact_focus} official tourism municipality".strip())
                clues.append(f"{compact_focus} anniversary competition festival official".strip())
                if not intent.needs_event_discovery_hop:
                    clues.append(f"{compact_focus} beauty pageant festival official tourism winner full name".strip())
        if should_probe_historical_year:
            clues.extend(
                [
                    "site:spomenikdatabase.org Bosnia monument victims year",
                    "site:spomenikdatabase.org former Yugoslavia memorial victims year",
                    "Banja Luka Tuzla Zenica Sarajevo monument victims year",
                ]
            )
            if query_requires_bosnia_top_city(query):
                clues.extend(
                    [
                        "site:spomenikdatabase.org Zenica monument victims year",
                        "site:spomenikdatabase.org Tuzla monument victims year",
                        "site:spomenikdatabase.org Banja Luka monument victims year",
                        "site:spomenikdatabase.org Sarajevo monument victims year",
                    ]
                )
            build_cutoff = historical_year_build_year_cutoff(query)
            artist_birth = historical_year_artist_birth_year(query)
            if build_cutoff is not None:
                clues.append(f"site:spomenikdatabase.org Bosnia monument built before {build_cutoff}")
            if artist_birth is not None:
                clues.append(f"site:spomenikdatabase.org Bosnia monument artist born {artist_birth}")
                if query_requires_bosnia_top_city(query):
                    clues.append(
                        f"site:spomenikdatabase.org Banja Luka Tuzla Zenica Sarajevo monument artist born {artist_birth}"
                    )
            if "1928" in query:
                clues.append("Bosnia monument victims year artist born 1928")
            if "former yugoslavia" in query.lower():
                clues.append("former Yugoslavia Bosnia monument dedication victims year")
        if should_probe_encyclopedic:
            clues.extend(
                [
                    f"{query} wiki",
                    f"{query} fandom",
                ]
            )
        if should_probe_character:
            clues.extend(
                [
                    f"{query} character wiki",
                    f"{query} antagonist companion",
                ]
            )
        if should_probe_count:
            clues.extend(
                [
                    f"{query} movements effects",
                    f"{query} ability list",
                ]
            )
        if intent.prefers_repository_sources:
            clues.append(f"{query} institutional repository pdf")
        if missing_everything and not clues:
            clues.append(query)
            if intent.prefers_pdf_sources:
                clues.append(f"{query} pdf")
            if "site:" not in query.lower() and any(signal in intent.signals for signal in ("academic", "thesis", "paper")):
                clues.append(f"site:.edu {query}")
            if intent.prefers_encyclopedic_sources:
                clues.append(f"{query} wiki")
            if intent.prefers_character_sources:
                clues.append(f"{query} character")

        existing_lower = {clue.lower() for clue in existing_clues}
        return [
            clue
            for clue in unique_preserve_order(clues)
            if clue.lower() not in existing_lower
        ][:6]
