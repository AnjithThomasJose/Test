import re
import unicodedata

BULLETS = "•●○■▪►▷▶❖✓✔✦–—"
SEP_LINE_FALLBACK = re.compile(r"^[^A-Za-z0-9]{3,}$")


def normalize(txt: str) -> str:
    s = unicodedata.normalize("NFKC", txt)
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = s.replace("\u00A0", " ")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("—", "-").replace("–", "-")

    trans = str.maketrans({ch: "-" for ch in BULLETS})
    s = s.translate(trans)

    out_lines = []
    for line in s.splitlines():
        ln = line.strip()
        # Collapse letter-spaced headings like "P R O F E S S I O N A L   E X P E R I E N C E"
        # into "PROFESSIONAL EXPERIENCE" (preserve space between words)
        spaced_groups = re.split(r"\s{2,}", ln)
        if len(spaced_groups) > 1:
            collapsed_groups = []
            for seg in spaced_groups:
                if re.fullmatch(r"(?:[A-Z](?:\s+[A-Z])+)", seg):
                    collapsed_groups.append(seg.replace(" ", ""))
                else:
                    collapsed_groups.append(seg)
            ln = " ".join(collapsed_groups)
        else:
            # If the whole line is a single spaced word
            if re.fullmatch(r"(?:[A-Z](?:\s+[A-Z])+)", ln):
                ln = ln.replace(" ", "")

        if re.match(SEP_LINE_FALLBACK, ln) and sum(c.isalnum() for c in ln) < max(1, int(0.1 * len(ln))):
            continue
        ln = re.sub(r"([=*_#\-])\1{2,}", "-", ln)
        out_lines.append(ln)
    s = "\n".join(out_lines)

    s = re.sub(r"\s{2,}", " ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    s = re.sub(r"-\n(?!\s*-|\s*$)", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


