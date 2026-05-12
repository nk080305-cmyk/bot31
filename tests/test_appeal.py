"""Smoke test for bot.handlers.appeal import."""


def test_module_importable():
    """Verify that bot.handlers.appeal can be imported without errors."""
    import bot.handlers.appeal  # noqa: F401 - side-effect import

    assert hasattr(bot.handlers.appeal, "router")
