"""Source-checkout entry point; a console script can also call cli.main."""

from .cli import main

raise SystemExit(main())
