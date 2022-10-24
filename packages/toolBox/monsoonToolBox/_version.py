
class Version:
    version_history = [
        ["0.0.1.0", "Init"],
    ]
    VERSION = version_history[-1][0]
    MAJOR, MINOR, PATCH, REVISION = VERSION.split(".")

    def version(self) -> str:
        return ".".join([self.MAJOR, self.MINOR, self.PATCH])

__version__ = Version().version()