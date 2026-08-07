"""Fetch species reference photographs from iNaturalist.

This is the one part of the toolkit that touches the network, so it is opt-in
and it never fails a report: anything it cannot fetch is recorded as a skipped
species and the report is written without that photograph.

Photographs are downloaded into the output package rather than hot-linked. A
package has to survive being archived and served offline, and hot-linking would
also disclose every reader's address to a third party.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .common import ToolError

TAXA_ENDPOINT = "https://api.inaturalist.org/v1/taxa"
TAXON_PAGE = "https://www.inaturalist.org/taxa/{taxon_id}"

USER_AGENT = (
    f"field-audio-tools/{__version__} (birdidpv; "
    "https://mantle-sound.org/software/)"
)

# Photographs carrying no licence are all-rights-reserved and are never
# downloaded. The rest are opt-out through --photo-licenses. ND is allowed by
# default because the file is stored verbatim, exactly as iNaturalist serves
# it, and no derivative is made.
DEFAULT_LICENSES = (
    "cc0",
    "cc-by",
    "cc-by-sa",
    "cc-by-nc",
    "cc-by-nc-sa",
    "cc-by-nd",
    "cc-by-nc-nd",
)

PHOTO_SIZES = ("square", "medium")

_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
class SpeciesPhoto:
    scientific_name: str
    taxon_id: int
    common_name: str | None
    file_name: str
    photo_id: int
    license_code: str
    attribution: str
    photographer: str | None
    source_url: str
    taxon_url: str
    wikipedia_url: str | None

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def _get_json(url: str, *, timeout: float, user_agent: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def resolve_taxon(
    scientific_name: str,
    *,
    timeout: float,
    user_agent: str = USER_AGENT,
) -> dict[str, Any] | None:
    """Return the taxon whose name matches exactly, or None.

    iNaturalist's search is fuzzy and will happily return a congener first:
    querying Pyrrhocorax pyrrhocorax puts Pyrrhocorax graculus at the top. A
    near miss here would caption the report with the wrong bird, so only an
    exact name match counts.
    """
    query = urllib.parse.urlencode(
        {"q": scientific_name, "rank": "species", "per_page": 20}
    )
    payload = _get_json(
        f"{TAXA_ENDPOINT}?{query}", timeout=timeout, user_agent=user_agent
    )
    wanted = scientific_name.strip().lower()
    for taxon in payload.get("results", []):
        if str(taxon.get("name", "")).strip().lower() == wanted:
            return taxon
    return None


def select_photo(
    taxon: dict[str, Any], allowed_licenses: Iterable[str]
) -> dict[str, Any] | None:
    allowed = {value.strip().lower() for value in allowed_licenses}
    photo = taxon.get("default_photo") or {}
    license_code = photo.get("license_code")
    if not license_code or str(license_code).lower() not in allowed:
        return None
    return photo


def _photo_url(photo: dict[str, Any], size: str) -> str | None:
    return photo.get(f"{size}_url") or photo.get("url")


def download_photo(
    url: str,
    destination_dir: Path,
    stem: str,
    *,
    timeout: float,
    user_agent: str = USER_AGENT,
) -> str:
    """Download one photograph verbatim and return its file name."""
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = (response.headers.get("Content-Type") or "").split(";")[0]
        data = response.read()
    suffix = _EXTENSIONS.get(content_type.strip().lower())
    if suffix is None:
        suffix = Path(urllib.parse.urlparse(url).path).suffix or ".jpg"
    destination_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{stem}{suffix}"
    (destination_dir / file_name).write_bytes(data)
    return file_name


def _stem(scientific_name: str) -> str:
    return scientific_name.strip().lower().replace(" ", "-")


def collect_species_photos(
    scientific_names: Iterable[str],
    destination_dir: Path,
    *,
    size: str = "medium",
    allowed_licenses: Iterable[str] = DEFAULT_LICENSES,
    timeout: float = 20.0,
    delay: float = 1.0,
    user_agent: str = USER_AGENT,
    progress: bool = False,
) -> tuple[list[SpeciesPhoto], list[tuple[str, str]]]:
    """Fetch one reference photograph per species.

    Returns the photographs collected and, for everything else, the species
    name paired with the reason it was skipped. Network trouble is a skip, not
    an error: the report is still worth writing without a picture.
    """
    if size not in PHOTO_SIZES:
        raise ToolError(f"--photo-size must be one of: {', '.join(PHOTO_SIZES)}")
    allowed = list(allowed_licenses)
    if not allowed:
        raise ToolError("--photo-licenses must list at least one licence")

    photos: list[SpeciesPhoto] = []
    skipped: list[tuple[str, str]] = []
    names = list(scientific_names)
    for index, name in enumerate(names):
        if index and delay:
            time.sleep(delay)
        if progress:
            print(f"  photo {index + 1}/{len(names)}: {name}", flush=True)
        try:
            taxon = resolve_taxon(name, timeout=timeout, user_agent=user_agent)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            skipped.append((name, f"lookup failed: {exc}"))
            continue
        if taxon is None:
            skipped.append((name, "no exact name match on iNaturalist"))
            continue
        photo = select_photo(taxon, allowed)
        if photo is None:
            code = (taxon.get("default_photo") or {}).get("license_code")
            reason = (
                "photograph is all rights reserved"
                if not code
                else f"photograph licence {code} is not in --photo-licenses"
            )
            skipped.append((name, reason))
            continue
        url = _photo_url(photo, size)
        if not url:
            skipped.append((name, "iNaturalist returned no image URL"))
            continue
        try:
            file_name = download_photo(
                url,
                destination_dir,
                _stem(name),
                timeout=timeout,
                user_agent=user_agent,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            skipped.append((name, f"download failed: {exc}"))
            continue
        photos.append(
            SpeciesPhoto(
                scientific_name=name,
                taxon_id=int(taxon["id"]),
                common_name=taxon.get("preferred_common_name"),
                file_name=file_name,
                photo_id=int(photo["id"]),
                license_code=str(photo["license_code"]),
                attribution=str(photo.get("attribution") or ""),
                photographer=photo.get("attribution_name"),
                source_url=url,
                taxon_url=TAXON_PAGE.format(taxon_id=taxon["id"]),
                wikipedia_url=taxon.get("wikipedia_url"),
            )
        )
    return photos, skipped


def write_credits(
    path: Path,
    photos: list[SpeciesPhoto],
    skipped: list[tuple[str, str]],
    *,
    photo_dir: str,
) -> None:
    """Record where every photograph came from and under what licence."""
    payload = {
        "source": "iNaturalist",
        "api": TAXA_ENDPOINT,
        "retrieved_by": {"tool": "birdidpv", "version": __version__},
        "photo_dir": photo_dir,
        "note": (
            "Each file is stored verbatim as iNaturalist served it. Licences "
            "and attribution belong to the individual photographers listed here."
        ),
        "photos": [photo.as_json() for photo in photos],
        "skipped": [{"scientific_name": name, "reason": reason} for name, reason in skipped],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
