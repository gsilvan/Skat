"""Optional local web interface; importing the engine never requires Flask."""


def create_app(**kwargs):
    from .app import create_app as factory

    return factory(**kwargs)


__all__ = ["create_app"]
