"""
PII privacy guard — Stage 4 of the query pipeline.

Runs AFTER SQL execution and BEFORE results are returned to the user.
Scans column names in the result DataFrame for PII patterns and masks
their values before they are displayed or stored in query history.

What gets masked:
  Detection is column-name-based (not content-based). This is intentional:
  - Faster than scanning cell contents
  - No false negatives from unexpected value formats
  - Consistent with how enterprise systems implement column-level security
  - Works correctly even when values are NULL or numeric IDs

  PII column patterns detected (case-insensitive substring match):
    email           → user@domain.com
    phone           → phone_number, phone_no, mobile
    national_id     → national_id, nid, ssn, tax_id
    passport        → passport_number, passport_no
    account_number  → account_number, account_no, acct_no
    sort_code       → sort_code, sort_no
    dob             → date_of_birth, dob, birth_date, birthdate
    credit_card     → credit_card, card_number, card_no, cvv
    address         → address_line1, address_line2 (NOT city/country)
    ip_address      → ip_address, ip_addr (session data)
    licence         → licence_number, license_number (advisor data)

What is NOT masked:
  - customer_id, account_id (they are INT identifiers — not PII in isolation)
  - city, country, postcode (geographic aggregation — low sensitivity)
  - name columns (first_name, last_name) — borderline; masked by default
    for a financial institution context (can be disabled via allow_names=True)

Masking strategy:
  Values are replaced with '***' — short enough to fit in any column width
  in the UI, but unambiguous that masking has occurred. NULL values are left
  as NULL (they carry no information regardless).
"""

import pandas as pd
from typing import Tuple, List, Set


# ── PII column pattern groups ──────────────────────────────────────────────────
# Each pattern is a substring that, if found in a column name (case-insensitive),
# flags that column as PII.

_EMAIL_PATTERNS = {"email", "e_mail", "emailaddress"}
_PHONE_PATTERNS = {"phone", "mobile", "tel", "telephone", "contact_no"}
_NATIONAL_PATTERNS = {
    "national_id",
    "nid",
    "ssn",
    "social_security",
    "tax_id",
    "national_insurance",
    "nin",
}
_PASSPORT_PATTERNS = {"passport"}
_ACCOUNT_PATTERNS = {"account_number", "account_no", "acct_no", "account_num"}
_SORT_PATTERNS = {"sort_code", "sort_no"}
_DOB_PATTERNS = {"date_of_birth", "dob", "birth_date", "birthdate", "dateofbirth"}
_CARD_PATTERNS = {"credit_card", "card_number", "card_no", "cvv", "cvc", "card_num"}
_ADDRESS_PATTERNS = {
    "address_line1",
    "address_line2",
    "address_line",
    "street_address",
    "street",
    "addr1",
    "addr2",
}
_IP_PATTERNS = {"ip_address", "ip_addr", "ipaddress"}
_LICENCE_PATTERNS = {
    "licence_number",
    "license_number",
    "licence_no",
    "license_no",
    "fca_number",
}
_NAME_PATTERNS = {
    "first_name",
    "last_name",
    "full_name",
    "surname",
    "firstname",
    "lastname",
    "fullname",
}
_DOCUMENT_PATTERNS = {"document_number", "doc_number", "doc_no", "document_no"}

# All patterns in one set for fast lookup
_ALL_PII_PATTERNS: Set[str] = (
    _EMAIL_PATTERNS
    | _PHONE_PATTERNS
    | _NATIONAL_PATTERNS
    | _PASSPORT_PATTERNS
    | _ACCOUNT_PATTERNS
    | _SORT_PATTERNS
    | _DOB_PATTERNS
    | _CARD_PATTERNS
    | _ADDRESS_PATTERNS
    | _IP_PATTERNS
    | _LICENCE_PATTERNS
    | _NAME_PATTERNS
    | _DOCUMENT_PATTERNS
)

MASK_VALUE = "***"


# ── PrivacyGuard ───────────────────────────────────────────────────────────────


class PrivacyGuard:
    """
    Column-level PII masking for DataFrame results.
    """

    def __init__(self, allow_names: bool = False):
        """
        Args:
            allow_names: If True, name columns (first_name, last_name, full_name)
                         are NOT masked. Default False for banking context where
                         names combined with account data constitute PII.
        """
        self.allow_names = allow_names

    def scan_and_mask(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Scan all column names in df for PII patterns and mask matching columns.

        Returns:
            (masked_df, masked_column_names)
            masked_df has PII columns replaced with MASK_VALUE.
            masked_column_names is the list of columns that were masked.
            If nothing was masked, returns (original_df, []).
        """
        if df is None or df.empty:
            return df, []

        pii_columns = self._identify_pii_columns(df.columns.tolist())

        if not pii_columns:
            return df, []

        # Copy so we do not mutate the original DataFrame
        masked_df = df.copy()
        for col in pii_columns:
            masked_df[col] = masked_df[col].apply(
                lambda v: MASK_VALUE if v is not None and pd.notna(v) else v
            )

        return masked_df, pii_columns

    def is_pii_column(self, column_name: str) -> bool:
        """Check if a single column name matches any PII pattern."""
        return bool(self._identify_pii_patterns(column_name.lower()))

    def get_safe_columns(self, df: pd.DataFrame) -> List[str]:
        """Return column names that are NOT PII — safe to display unmasked."""
        all_cols = df.columns.tolist()
        pii = set(self._identify_pii_columns(all_cols))
        return [c for c in all_cols if c not in pii]

    def audit_dataframe(self, df: pd.DataFrame) -> dict:
        """
        Return a structured audit of which columns are PII and which are safe.
        Used for logging and for the source-transparency panel in the UI.
        """
        all_cols = df.columns.tolist()
        pii_cols = self._identify_pii_columns(all_cols)
        safe_cols = [c for c in all_cols if c not in pii_cols]

        return {
            "total_columns": len(all_cols),
            "pii_columns": pii_cols,
            "safe_columns": safe_cols,
            "pii_count": len(pii_cols),
            "mask_value": MASK_VALUE,
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _identify_pii_columns(self, column_names: List[str]) -> List[str]:
        """Return the subset of column_names that match PII patterns."""
        pii = []
        for col in column_names:
            if self._identify_pii_patterns(col.lower()):
                pii.append(col)
        return pii

    def _identify_pii_patterns(self, col_lower: str) -> Set[str]:
        """
        Return the set of PII pattern groups that match this column name.
        A column name is PII if ANY pattern from ANY group is a substring of it.
        """
        matched = set()

        # If allow_names is True, skip name patterns
        active_patterns = _ALL_PII_PATTERNS
        if self.allow_names:
            active_patterns = _ALL_PII_PATTERNS - _NAME_PATTERNS

        # Direct exact match against the full pattern set first (fast path)
        if col_lower in active_patterns:
            matched.add(col_lower)
            return matched

        # Substring match: col_lower contains any pattern
        for pattern in active_patterns:
            if pattern in col_lower:
                matched.add(pattern)

        # Reverse substring: pattern contains col_lower
        # Catches cases like col_name='dob' matching pattern 'date_of_birth'
        for pattern in active_patterns:
            if col_lower in pattern and len(col_lower) >= 3:
                matched.add(pattern)

        return matched
