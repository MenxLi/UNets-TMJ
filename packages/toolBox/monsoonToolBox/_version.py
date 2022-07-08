# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: _version.py                                            | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #

class Version:
    version_history = [
        ["0.0.1.0", "Init"],
    ]
    VERSION = version_history[-1][0]
    MAJOR, MINOR, PATCH, REVISION = VERSION.split(".")

    def version(self) -> str:
        return ".".join([self.MAJOR, self.MINOR, self.PATCH])

__version__ = Version().version()