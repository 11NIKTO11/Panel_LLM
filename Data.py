# Canonical names used across VotingResult and data mappings
# Always import and use these instead of string literals.
VOTED = "Voted"
NOT_VOTED = "Not Voted"

ANO = "ANO 2011"
SPOLU = "Koalice Spolu (ODS, TOP 09, KDU-ČSL)"
PIRSTAN = "Koalice PIRÁTI a STAROSTOVÉ"
KSCM = "Komunistická strana Čech a Moravy (KSČM)"
SPD = "Svoboda a přímá demokracie – Tomio Okamura (SPD)"
CSSD = "Česká strana sociálně demokratická (ČSSD)"
TSS = "Trikolóra, Svobodní a Soukromníci"
PRISAHA = "Přísaha Roberta Šlachty"
JINA_STRANA = "Jiná strana"

# Explicit party column order used in CSVs/Series
PARTY_COLUMNS = [
    ANO, SPOLU, PIRSTAN, KSCM, SPD, CSSD, TSS, PRISAHA, JINA_STRANA
]

# Convenience loader for actual election results CSV
import os
import pandas as pd
from typing import Optional

def get_actual_results() -> pd.DataFrame:
    csv_path = os.path.join("data", "election_data.csv")
    return pd.read_csv(csv_path, encoding="utf-8-sig")