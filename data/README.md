# Local data directory

This directory is intentionally excluded from the public repository except for this file.

The following data is created or kept locally:

- `raw/`: licensed K-IFRS source documents
- `processed/`: parsed paragraphs, blocks, tables, footnotes, and references
- `chunks/`: retrieval chunks derived from the standards
- `embeddings/`: OpenAI embedding cache and manifests
- `semantic/`: Concept and MENTIONS semantic KG exports
- `validation/`: local parsing and mapping review artifacts

Do not publish these directories without separately confirming that redistribution is permitted.
The public repository should contain the processing code, schemas, configuration examples, and documentation only.
