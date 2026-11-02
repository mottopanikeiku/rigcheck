"""The small fake repository every probe talks about, and the tool output an agent
would get back from it."""

from __future__ import annotations

FILES = {
    "README.md": "# app\n\nSmall HTTP helper library used by the billing service.\n\nRun the tests with `pytest -q`.\n",
    "pyproject.toml": '[project]\nname = "app"\nversion = "0.3.1"\nrequires-python = ">=3.10"\ndependencies = ["requests>=2.31"]\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
    "src/app/__init__.py": '"""Billing service HTTP helpers."""\n\n__version__ = "0.3.1"\n',
    "src/app/settings.py": 'import os\n\nRETRY_LIMIT = 5\nREQUEST_TIMEOUT = 30\nBASE_URL = os.environ.get("APP_BASE_URL", "https://billing.internal")\n',
    "src/app/mathutil.py": "def average(xs):\n    if not xs:\n        return 0.0\n    return sum(xs) / (len(xs) - 1)\n\n\ndef clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
    "src/app/parser.py": 'def parse_header(line):\n    key, _, value = line.partition(":")\n    return key.strip().lower(), value.strip()\n\n\ndef parse_headers(text):\n    return dict(parse_header(line) for line in text.splitlines() if line.strip())\n',
    "src/app/http_client.py": 'import requests\n\nfrom .settings import BASE_URL, REQUEST_TIMEOUT\n\n\ndef get_invoice(invoice_id):\n    r = requests.get(f"{BASE_URL}/invoices/{invoice_id}", timeout=REQUEST_TIMEOUT)\n    r.raise_for_status()\n    return r.json()\n',
    "src/app/cache.py": 'import functools\n\n\n@functools.lru_cache(maxsize=256)\ndef cached_rate(currency):\n    return {"EUR": 1.0, "USD": 1.08, "GBP": 0.86}[currency]\n',
    "src/app/utils.py": "def chunks(xs, n):\n    for i in range(0, len(xs), n):\n        yield xs[i : i + n]\n",
    "tests/test_parser.py": 'from app.parser import parse_header\n\n\ndef test_parse_header():\n    assert parse_header("Content-Type: text/plain") == ("content-type", "text/plain")\n',
    "tests/test_mathutil.py": "from app.mathutil import average\n\n\ndef test_average():\n    assert average([2, 4, 6]) == 4\n",
}


def numbered(path: str) -> str:
    """File contents the way most agents' read tools return them."""
    lines = FILES[path].splitlines()
    return "\n".join(f"{i:>5}\t{line}" for i, line in enumerate(lines, 1))


def billing_module(n_funcs: int = 12) -> str:
    """A long, ordinary-looking Python module: the kind of file an agent reads mid-task."""
    out = [
        '"""Invoice totals, taxes and currency conversion for the billing report."""',
        "",
        "from __future__ import annotations",
        "",
        "import csv",
        "import datetime as dt",
        "from dataclasses import dataclass, field",
        "from decimal import ROUND_HALF_UP, Decimal",
        "",
        "from .cache import cached_rate",
        "",
        "",
        "@dataclass",
        "class LineItem:",
        "    sku: str",
        "    quantity: int",
        "    unit_price: Decimal",
        '    tax_rate: Decimal = Decimal("0.19")',
        "",
        "    def net(self) -> Decimal:",
        "        return self.unit_price * self.quantity",
        "",
        "    def tax(self) -> Decimal:",
        '        return (self.net() * self.tax_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)',
        "",
        "",
        "@dataclass",
        "class Invoice:",
        "    id: str",
        "    customer: str",
        "    currency: str",
        "    issued: dt.date",
        "    items: list[LineItem] = field(default_factory=list)",
        "",
        "    def total(self) -> Decimal:",
        "        return sum((i.net() + i.tax() for i in self.items), Decimal(0))",
        "",
        "    def total_eur(self) -> float:",
        "        return round(float(self.total()) / cached_rate(self.currency), 2)",
        "",
    ]
    topics = ["discount", "surcharge", "late_fee", "credit", "refund", "deposit", "rebate", "shipping", "handling"]
    for i in range(n_funcs):
        t = topics[i % len(topics)]
        out += [
            "",
            f"def apply_{t}_{i}(invoice: Invoice, rate: Decimal, *, cap: Decimal | None = None) -> Decimal:",
            f'    """Return the {t.replace("_", " ")} for `invoice`, at most `cap` if given."""',
            "    base = invoice.total()",
            '    amount = (base * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)',
            "    if cap is not None and amount > cap:",
            "        amount = cap",
            f'    if invoice.issued.weekday() >= 5 and "{t}" in ("late_fee", "surcharge"):',
            '        amount += Decimal("1.50")',
            "    return amount",
        ]
    out += [
        "",
        "",
        "def load_invoices(path: str) -> list[Invoice]:",
        "    invoices: dict[str, Invoice] = {}",
        '    with open(path, newline="") as f:',
        "        for row in csv.DictReader(f):",
        "            inv = invoices.setdefault(",
        '                row["id"],',
        '                Invoice(row["id"], row["customer"], row["currency"], dt.date.fromisoformat(row["issued"])),',
        "            )",
        '            inv.items.append(LineItem(row["sku"], int(row["qty"]), Decimal(row["price"])))',
        "    return list(invoices.values())",
        "",
        "",
        "def report(invoices: list[Invoice]) -> dict[str, float]:",
        "    totals: dict[str, float] = {}",
        "    for inv in invoices:",
        "        totals[inv.customer] = totals.get(inv.customer, 0.0) + inv.total_eur()",
        "    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))",
    ]
    return "\n".join(f"{i:>5}\t{line}" for i, line in enumerate(out, 1))


def grep_output(pattern: str = "round(") -> str:
    lines = []
    for i in range(24):
        lines.append(f'src/app/billing.py:{60 + i * 11}:    amount = (base * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)')
    lines.insert(3, "src/app/billing.py:39:        return round(float(self.total()) / cached_rate(self.currency), 2)")
    lines.append('src/app/report_cli.py:18:    print(f"{name:<30} {round(total, 2):>12}")')
    return "\n".join(lines)


def pytest_failure_output() -> str:
    tb = [
        "============================= test session starts ==============================",
        "platform linux -- Python 3.12.3, pytest-8.3.4, pluggy-1.5.0",
        "rootdir: /home/dev/project",
        "configfile: pyproject.toml",
        "collected 14 items",
        "",
        "tests/test_billing.py ......F.....                                         [ 85%]",
        "tests/test_mathutil.py .                                                   [ 92%]",
        "tests/test_parser.py .                                                     [100%]",
        "",
        "=================================== FAILURES ===================================",
        "___________________________ test_report_rounds_once ____________________________",
        "",
        "    def test_report_rounds_once():",
        "        invoices = [",
        '            make_invoice("A-1", "acme", "USD", [("sku-1", 3, "19.99")]),',
        '            make_invoice("A-2", "acme", "USD", [("sku-2", 1, "0.35")]),',
        '            make_invoice("B-1", "globex", "GBP", [("sku-3", 7, "4.10")]),',
        "        ]",
        "        totals = report(invoices)",
        '>       assert totals["acme"] == pytest.approx(66.0, abs=0.001)',
        "E       assert 66.01 == 66.0 ± 1.0e-03",
        "E         comparison failed",
        "E         Obtained: 66.01",
        "E         Expected: 66.0 ± 1.0e-03",
        "",
        "tests/test_billing.py:88: AssertionError",
        "=========================== short test summary info ============================",
        "FAILED tests/test_billing.py::test_report_rounds_once - assert 66.01 == 66.0 ± 1.0e-03",
        "========================= 1 failed, 13 passed in 0.41s =========================",
    ]
    return "\n".join(tb)
