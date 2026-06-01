"""
Fetch nucleotide (CDS) sequences encoding proteins for a list of PDB IDs.

Workflow:
  1. Read PDB IDs from a .txt file (one per line).
  2. For each PDB ID, query the RCSB PDB REST API to get UniProt accession(s)
     linked to that structure.
  3. Use the UniProt accession to retrieve the canonical CDS (coding sequence)
     from the NCBI Entrez API via the linked nucleotide records.
  4. Write all results to a multi-entry FASTA file.

Usage:
    python fetch_nn.py pdbids_tpchackathon.txt
    python fetch_nn.py pdbids_tpchackathon.txt --out nn_from_pdbid.fasta --email xlian@anl.gov
"""

import argparse
import sys
import time
import json
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

# ── helpers ─────────────────────────────────────────────────────────────────

def fetch_url(url: str, retries: int = 3, delay: float = 1.0) -> bytes:
    """GET a URL with simple retry logic."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            raise


# ── Step 1 – PDB → UniProt accessions ───────────────────────────────────────

def pdb_to_uniprot(pdb_id: str) -> list[str]:
    """
    Use the RCSB Data API to retrieve UniProt accessions for a PDB entry.
    Returns a (possibly empty) list of accession strings.
    """
    url = (
        f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
    )
    try:
        raw = fetch_url(url)
    except urllib.error.HTTPError:
        return []

    data = json.loads(raw)

    accessions: list[str] = []
    # polymer_entities → rcsb_polymer_entity_container_identifiers → uniprot_ids
    entities = data.get("rcsb_entry_container_identifiers", {}).get(
        "polymer_entity_ids", []
    )
    for eid in entities:
        ent_url = (
            f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}/{eid}"
        )
        try:
            ent_raw = fetch_url(ent_url)
        except Exception:
            continue
        ent_data = json.loads(ent_raw)
        ups = (
            ent_data.get("rcsb_polymer_entity_container_identifiers", {})
            .get("uniprot_ids", [])
        )
        for acc in ups:
            if acc not in accessions:
                accessions.append(acc)

    return accessions


# ── Step 2 – UniProt → NCBI nucleotide ID ────────────────────────────────────

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def uniprot_to_nuccore_ids(uniprot_acc: str, email: str) -> list[str]:
    """
    Search NCBI Nucleotide (nuccore) for records linked to a UniProt accession.
    Returns a list of NCBI UIDs (strings).
    """
    query = urllib.parse.quote(f"{uniprot_acc}[Accession] AND mRNA[Filter]")
    url = (
        f"{NCBI_BASE}/esearch.fcgi"
        f"?db=nuccore&term={query}&retmax=5&retmode=json&email={email}"
    )
    try:
        raw = fetch_url(url)
    except Exception:
        return []

    data = json.loads(raw)
    ids = data.get("esearchresult", {}).get("idlist", [])

    # Fallback: search by UniProt accession as cross-reference
    if not ids:
        query2 = urllib.parse.quote(f"{uniprot_acc}[Keyword] AND CDS[Feature Key]")
        url2 = (
            f"{NCBI_BASE}/esearch.fcgi"
            f"?db=nuccore&term={query2}&retmax=5&retmode=json&email={email}"
        )
        try:
            raw2 = fetch_url(url2)
            ids = json.loads(raw2).get("esearchresult", {}).get("idlist", [])
        except Exception:
            pass

    return ids


# ── Step 3 – Fetch FASTA from NCBI ───────────────────────────────────────────

def fetch_fasta_from_ncbi(uid: str, email: str) -> str | None:
    """Fetch the nucleotide FASTA for a single NCBI UID."""
    url = (
        f"{NCBI_BASE}/efetch.fcgi"
        f"?db=nuccore&id={uid}&rettype=fasta&retmode=text&email={email}"
    )
    try:
        raw = fetch_url(url)
        text = raw.decode("utf-8").strip()
        return text if text.startswith(">") else None
    except Exception:
        return None


# ── Alternative: UniProt → ENA CDS FASTA ─────────────────────────────────────

def fetch_cds_from_uniprot_ena(uniprot_acc: str) -> str | None:
    """
    Fetch CDS FASTA directly from UniProt's ENA cross-reference.
    Returns a FASTA string or None.
    """
    # 1. Get the UniProt entry JSON to find ENA/EMBL cross-refs
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_acc}.json"
    try:
        raw = fetch_url(url)
    except Exception:
        return None

    data = json.loads(raw)
    ena_ids: list[str] = []
    for xref in data.get("uniProtKBCrossReferences", []):
        if xref.get("database") in ("EMBL", "ENA"):
            for prop in xref.get("properties", []):
                if prop.get("key") == "ProteinId" and prop.get("value", "-") != "-":
                    protein_id = prop["value"]
                    # the genomic nucleotide accession
                    nuc_acc = xref.get("id", "")
                    if nuc_acc:
                        ena_ids.append(nuc_acc)

    if not ena_ids:
        return None

    # Fetch FASTA from ENA for the first hit
    for ena_id in ena_ids[:3]:
        url2 = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{ena_id}?download=false"
        try:
            raw2 = fetch_url(url2)
            text = raw2.decode("utf-8").strip()
            if text.startswith(">"):
                return text
        except Exception:
            continue

    return None


# ── main pipeline ─────────────────────────────────────────────────────────────

def process_pdb_list(pdb_ids: list[str], out_path: Path, email: str) -> None:
    results: list[str] = []
    not_found: list[str] = []

    for pdb_id in pdb_ids:
        pdb_id = pdb_id.strip().upper()
        if not pdb_id:
            continue

        print(f"\n[{pdb_id}] Fetching UniProt accessions …")
        uniprot_ids = pdb_to_uniprot(pdb_id)

        if not uniprot_ids:
            print(f"  ✗ No UniProt mapping found for {pdb_id}")
            not_found.append(pdb_id)
            continue

        print(f"  UniProt IDs: {', '.join(uniprot_ids)}")
        found_any = False

        for acc in uniprot_ids:
            print(f"  [{acc}] Trying ENA/EMBL CDS via UniProt …")
            fasta = fetch_cds_from_uniprot_ena(acc)
            time.sleep(0.4)  # be polite to EBI

            if not fasta:
                print(f"  [{acc}] Trying NCBI nuccore …")
                uid_list = uniprot_to_nuccore_ids(acc, email)
                time.sleep(0.35)
                for uid in uid_list[:2]:
                    fasta = fetch_fasta_from_ncbi(uid, email)
                    time.sleep(0.35)
                    if fasta:
                        break

            if fasta:
                # Prepend a helpful comment line
                header_tag = f"PDB:{pdb_id}|UniProt:{acc}"
                lines = fasta.splitlines()
                lines[0] = lines[0] + f" [{header_tag}]"
                results.append("\n".join(lines))
                print(f"  ✓ Sequence retrieved for {acc}")
                found_any = True
            else:
                print(f"  ✗ No nucleotide sequence found for {acc}")

        if not found_any:
            not_found.append(pdb_id)

    # Write FASTA output
    if results:
        out_path.write_text("\n\n".join(results) + "\n")
        print(f"\n✅  Wrote {len(results)} sequence(s) to {out_path}")
    else:
        print("\n⚠️  No sequences retrieved.")

    if not_found:
        print(f"⚠️  No nucleotide sequence found for: {', '.join(not_found)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch nucleotide (CDS) sequences for PDB IDs."
    )
    parser.add_argument(
        "input",
        help="Path to a .txt file with one PDB ID per line.",
    )
    parser.add_argument(
        "--out",
        default="nucleotide_sequences.fasta",
        help="Output FASTA file (default: nucleotide_sequences.fasta).",
    )
    parser.add_argument(
        "--email",
        default="user@example.com",
        help="E-mail for NCBI Entrez (required by NCBI policy).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: input file '{input_path}' not found.")

    pdb_ids = [line.strip() for line in input_path.read_text().splitlines() if line.strip()]
    if not pdb_ids:
        sys.exit("Error: no PDB IDs found in the input file.")

    print(f"Loaded {len(pdb_ids)} PDB ID(s): {', '.join(pdb_ids)}")
    process_pdb_list(pdb_ids, Path(args.out), args.email)


if __name__ == "__main__":
    main()