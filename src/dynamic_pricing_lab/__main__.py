"""Launch the Streamlit dashboard: ``python -m dynamic_pricing_lab``.

Also installed as the ``dynamic-pricing-lab`` console script. Extra
arguments are handed to ``streamlit run``, so for example
``dynamic-pricing-lab --server.port 8600`` works.
"""

import sys
from pathlib import Path

# The site theme (.streamlit/config.toml), passed as flags so the
# installed dashboard looks right from any working directory. Flags
# given on the command line come later and win.
_FLAGS = [
    "--theme.base=light",
    "--theme.primaryColor=#1e3a5f",
    "--theme.backgroundColor=#fafaf7",
    "--theme.secondaryBackgroundColor=#f2f1ec",
    "--theme.textColor=#1a1a1a",
    "--client.toolbarMode=viewer",
    "--browser.gatherUsageStats=false",
]


def main() -> None:
    try:
        from streamlit.web import cli as stcli
    except ModuleNotFoundError:
        sys.exit(
            "The dashboard needs Streamlit, which is an optional extra.\n"
            'Install it with:  pip install "dynamic-pricing-lab[dashboard]"'
        )
    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), *_FLAGS, *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
