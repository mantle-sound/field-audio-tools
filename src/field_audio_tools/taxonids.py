"""Cross-reference detected species to the taxonomic databases.

A detection is only useful if it can be carried somewhere else: to a checklist,
an occurrence database, a conservation assessment. Wikidata already holds those
identifiers for practically every bird, so this asks Wikidata rather than
maintaining a mapping.

Species are matched by iNaturalist taxon ID, never by name. A name search on
either service will happily return a congener — Pyrrhocorax pyrrhocorax finds
Pyrrhocorax graculus first — and an identifier attached to the wrong bird is
worse than no identifier at all.

The Wikidata SPARQL endpoint is not used. It is aggressively rate-limited and
periodically unavailable; the Action API is neither, and an exact statement
search is all this needs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .common import ToolError
from .inaturalist import USER_AGENT, resolve_taxon

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
WIKIDATA_ITEM = "https://www.wikidata.org/wiki/{qid}"

# The iNaturalist taxon ID property, which is how a species is pinned down.
INATURALIST_PROPERTY = "P3151"

# Curated for birds and biodiversity work. Every template below is the
# formatter URL Wikidata itself publishes for that property (P1630), checked
# against a live request rather than written from memory. Wikidata carries dozens more per
# species; those are kept in the JSON, and only these are meant for display.
# (property, label, URL template)
DISPLAY_IDENTIFIERS: tuple[tuple[str, str, str], ...] = (
    ("P2026", "Avibase", "https://avibase.bsc-eoc.org/species.jsp?avibaseid={id}"),
    ("P3444", "eBird", "https://ebird.org/species/{id}"),
    ("P846", "GBIF", "https://www.gbif.org/species/{id}"),
    ("P815", "ITIS", "https://www.itis.gov/servlet/SingleRpt/SingleRpt?search_topic=TSN&search_value={id}"),
    # /details/ is the form Wikidata publishes and the one that works: it
    # redirects to the current assessment. /species/{id}/0 is a 404.
    ("P627", "IUCN", "https://www.iucnredlist.org/details/{id}/0"),
    ("P830", "EOL", "https://eol.org/pages/{id}"),
    ("P3151", "iNaturalist", "https://www.inaturalist.org/taxa/{id}"),
)


@dataclass(frozen=True)
class TaxonIdentifiers:
    scientific_name: str
    common_name: str | None
    inaturalist_id: int
    wikidata_id: str
    identifiers: dict[str, str] = field(default_factory=dict)

    def links(self) -> list[tuple[str, str]]:
        """Label and URL for each curated identifier this species has."""
        found = [
            (label, template.format(id=self.identifiers[prop]))
            for prop, label, template in DISPLAY_IDENTIFIERS
            if prop in self.identifiers
        ]
        found.append(("Wikidata", WIKIDATA_ITEM.format(qid=self.wikidata_id)))
        return found

    def as_json(self) -> dict[str, Any]:
        return {
            "scientific_name": self.scientific_name,
            "common_name": self.common_name,
            "inaturalist_id": self.inaturalist_id,
            "wikidata_id": self.wikidata_id,
            "links": [{"label": l, "url": u} for l, u in self.links()],
            "identifiers": self.identifiers,
        }


def _get(url: str, *, timeout: float, user_agent: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def find_wikidata_item(
    inaturalist_id: int, *, timeout: float, user_agent: str = USER_AGENT
) -> str | None:
    """The Wikidata item whose iNaturalist taxon ID is exactly this one."""
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": f"haswbstatement:{INATURALIST_PROPERTY}={inaturalist_id}",
            "srlimit": 1,
            "format": "json",
        }
    )
    payload = _get(f"{WIKIDATA_API}?{query}", timeout=timeout, user_agent=user_agent)
    hits = payload.get("query", {}).get("search") or []
    return hits[0]["title"] if hits else None


def read_identifiers(
    qid: str, *, timeout: float, user_agent: str = USER_AGENT
) -> dict[str, str]:
    """Every external identifier on a Wikidata item, keyed by property."""
    payload = _get(
        WIKIDATA_ENTITY.format(qid=qid), timeout=timeout, user_agent=user_agent
    )
    claims = payload["entities"][qid]["claims"]
    identifiers: dict[str, str] = {}
    for prop, statements in claims.items():
        snak = statements[0].get("mainsnak", {})
        if snak.get("datatype") != "external-id":
            continue
        value = snak.get("datavalue", {}).get("value")
        if isinstance(value, str):
            identifiers[prop] = value
    return identifiers


def collect_taxon_identifiers(
    species: Iterable[tuple[str, str | None, int | None]],
    *,
    timeout: float = 20.0,
    delay: float = 0.5,
    user_agent: str = USER_AGENT,
    progress: bool = False,
) -> tuple[list[TaxonIdentifiers], list[tuple[str, str]]]:
    """Resolve identifiers for each species.

    Each entry is (scientific name, common name, iNaturalist taxon ID or None).
    A missing iNaturalist ID is resolved first, by exact name match. Anything
    that cannot be resolved is reported as skipped, never raised: identifiers
    are a convenience and must not cost anyone a report.
    """
    resolved: list[TaxonIdentifiers] = []
    skipped: list[tuple[str, str]] = []
    entries = list(species)
    for index, (scientific_name, common_name, inaturalist_id) in enumerate(entries):
        if index and delay:
            time.sleep(delay)
        if progress:
            print(f"  identifiers {index + 1}/{len(entries)}: {scientific_name}")
        try:
            if inaturalist_id is None:
                taxon = resolve_taxon(
                    scientific_name, timeout=timeout, user_agent=user_agent
                )
                if taxon is None:
                    skipped.append(
                        (scientific_name, "no exact name match on iNaturalist")
                    )
                    continue
                inaturalist_id = int(taxon["id"])
                common_name = common_name or taxon.get("preferred_common_name")
            qid = find_wikidata_item(
                inaturalist_id, timeout=timeout, user_agent=user_agent
            )
            if qid is None:
                skipped.append(
                    (
                        scientific_name,
                        f"no Wikidata item carries iNaturalist ID {inaturalist_id}",
                    )
                )
                continue
            identifiers = read_identifiers(
                qid, timeout=timeout, user_agent=user_agent
            )
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            OSError,
        ) as exc:
            skipped.append((scientific_name, f"lookup failed: {exc}"))
            continue
        resolved.append(
            TaxonIdentifiers(
                scientific_name=scientific_name,
                common_name=common_name,
                inaturalist_id=int(inaturalist_id),
                wikidata_id=qid,
                identifiers=identifiers,
            )
        )
    return resolved, skipped


def write_taxon_ids(
    path: Path,
    resolved: list[TaxonIdentifiers],
    skipped: list[tuple[str, str]],
) -> None:
    payload = {
        "source": "Wikidata",
        "api": WIKIDATA_API,
        "matched_on": (
            f"iNaturalist taxon ID ({INATURALIST_PROPERTY}), never on name"
        ),
        "retrieved_by": {"tool": "birdidpv", "version": __version__},
        "displayed": [label for _, label, _ in DISPLAY_IDENTIFIERS] + ["Wikidata"],
        "note": (
            "Wikidata content is CC0. The databases these identifiers point at "
            "carry their own terms."
        ),
        "species": [item.as_json() for item in resolved],
        "skipped": [
            {"scientific_name": name, "reason": reason} for name, reason in skipped
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
