from collections import defaultdict

import click


class CategorizedCommand(click.Command):
    """Click Command with support for categorized parameters."""

    def __init__(self, category_order, *args, **kwargs):
        # Set default context settings
        kwargs["context_settings"] = {
            "show_default": True,
            "max_content_width": 120,
            "help_option_names": ["-h", "--help"],
        }
        super().__init__(*args, **kwargs)

        # Get categories to use as section headers in the help page
        self.category_order = category_order + ["Other"]

    def format_help(self, ctx, formatter):
        """Format help using categorized display."""
        format_categorized_help(self, ctx, formatter, self.category_order)


def format_categorized_help(command, ctx, formatter, category_order: list):
    """Format help output with parameters grouped by category using Click's default formatting."""

    # Use Click's default usage and description formatting
    command.format_usage(ctx, formatter)
    if command.help:
        formatter.indent()
        formatter.write_paragraph()
        formatter.write_text(command.help)
        formatter.dedent()

    # Group parameters by category
    categories = defaultdict(list)
    for param in command.params:
        if isinstance(param, click.Argument):
            continue
        category = getattr(param, "category", "Other")
        categories[category].append(param)

    # Collect all help records first to calculate consistent spacing
    all_rows = []
    category_sections = []

    for category in category_order:
        params = categories.get(category, [])
        if not params:
            continue

        rows = []
        for param in params:
            rv = param.get_help_record(ctx)
            rows.append(rv)

        if rows:
            section_name = f"{category} Options"
            category_sections.append((section_name, rows))
            all_rows.extend(rows)

    # Calculate the maximum width needed across all categories
    # And print each parameter description with consistent spacing
    if all_rows:
        max_width = max(len(row[0]) for row in all_rows)
        for section_name, rows in category_sections:
            with formatter.section(section_name):
                for parameter, docstring in rows:
                    formatter.write_text(f"{parameter:<{max_width}}  {docstring}")


class CategorizedOption(click.Option):
    """Click Option with category support for grouped help display."""

    def __init__(self, *args, category=None, **kwargs):
        self.category = category or "Other"
        super().__init__(*args, **kwargs)


def categorized_option(*param_decls, category=None, **kwargs):
    """Decorator to add a categorized option to a command."""

    def decorator(f):
        if not hasattr(f, "__click_params__"):
            f.__click_params__ = []
        f.__click_params__.append(
            CategorizedOption(param_decls, category=category, **kwargs)
        )
        return f

    return decorator
