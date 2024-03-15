try:
    import gevent.monkey
    gevent.monkey.patch_all()
except ImportError:
    pass
