"""The scales a TAT-QA answer can carry, as the official scorer names them.

Kept apart from fingraph.eval.scorer so that the arms and the API can use it without
importing the official scorer, which pulls in pandas and SciPy (about 90 MB).
"""

SCALES = ("", "thousand", "million", "billion", "percent")
