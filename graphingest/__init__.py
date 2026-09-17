"""Ingest a folder of PDFs into a causal graph.

Point the pipeline at a directory containing PDFs and an RIS export and it
runs four stages, each usable on its own:

    graphingest.ris       PDFs + RIS        -> corpus manifest
    graphingest.convert   manifest          -> markdown articles
    graphingest.annotate  markdown          -> one causal graph per document
    graphingest.merge     document graphs   -> one graph, merged into an existing one

``graphingest.run`` chains all four. ``graphingest.validate`` checks any graph
against the schema.

Nothing here hardcodes a schema or a provider. The LinkML file passed as
``--schema`` drives the prompt, the output constraint, the normalizer and the
validator together; ``config/pipeline.yaml`` decides where inference runs.
"""

__version__ = "0.1.0"
