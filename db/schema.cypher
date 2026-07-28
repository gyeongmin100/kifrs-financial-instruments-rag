// K-IFRS Financial Instruments QA - Neo4j 5.x schema
// Re-runnable: every schema statement uses IF NOT EXISTS.
// text-embedding-3-large uses 3072 dimensions here. If that dimension changes,
// drop and recreate chunk_embedding_vector after rebuilding all embeddings.

// Unique identifiers
CREATE CONSTRAINT standard_id_unique IF NOT EXISTS
FOR (n:Standard) REQUIRE n.standard_id IS UNIQUE;

CREATE CONSTRAINT zone_id_unique IF NOT EXISTS
FOR (n:Zone) REQUIRE n.zone_id IS UNIQUE;

CREATE CONSTRAINT section_id_unique IF NOT EXISTS
FOR (n:Section) REQUIRE n.section_id IS UNIQUE;

CREATE CONSTRAINT paragraph_id_unique IF NOT EXISTS
FOR (n:Paragraph) REQUIRE n.paragraph_id IS UNIQUE;

CREATE CONSTRAINT subparagraph_id_unique IF NOT EXISTS
FOR (n:Subparagraph) REQUIRE n.subparagraph_id IS UNIQUE;

CREATE CONSTRAINT block_id_unique IF NOT EXISTS
FOR (n:Block) REQUIRE n.block_id IS UNIQUE;

CREATE CONSTRAINT table_id_unique IF NOT EXISTS
FOR (n:Table) REQUIRE n.table_id IS UNIQUE;

CREATE CONSTRAINT footnote_id_unique IF NOT EXISTS
FOR (n:Footnote) REQUIRE n.footnote_id IS UNIQUE;

CREATE CONSTRAINT pdf_page_id_unique IF NOT EXISTS
FOR (n:PdfPage) REQUIRE n.page_id IS UNIQUE;

CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE;

CREATE CONSTRAINT external_standard_id_unique IF NOT EXISTS
FOR (n:ExternalStandard) REQUIRE n.external_standard_id IS UNIQUE;

CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
FOR (n:Concept) REQUIRE n.concept_id IS UNIQUE;

CREATE CONSTRAINT rule_id_unique IF NOT EXISTS
FOR (n:Rule) REQUIRE n.rule_id IS UNIQUE;

CREATE CONSTRAINT condition_id_unique IF NOT EXISTS
FOR (n:Condition) REQUIRE n.condition_id IS UNIQUE;

CREATE CONSTRAINT exception_id_unique IF NOT EXISTS
FOR (n:Exception) REQUIRE n.exception_id IS UNIQUE;

CREATE CONSTRAINT example_id_unique IF NOT EXISTS
FOR (n:Example) REQUIRE n.example_id IS UNIQUE;

// Exact lookup and filtering indexes
CREATE INDEX paragraph_standard_number IF NOT EXISTS
FOR (n:Paragraph) ON (n.standard_id, n.number);

CREATE INDEX paragraph_zone_order IF NOT EXISTS
FOR (n:Paragraph) ON (n.standard_id, n.zone, n.document_order);

CREATE INDEX block_parent_order IF NOT EXISTS
FOR (n:Block) ON (n.parent_paragraph_id, n.document_order);

CREATE INDEX section_path_key IF NOT EXISTS
FOR (n:Section) ON (n.standard_id, n.path_key);

CREATE INDEX pdf_page_lookup IF NOT EXISTS
FOR (n:PdfPage) ON (n.standard_id, n.pdf_page);

CREATE INDEX chunk_search_filter IF NOT EXISTS
FOR (n:Chunk) ON (n.searchable, n.inactive, n.standard_id);

CREATE INDEX chunk_zone_priority IF NOT EXISTS
FOR (n:Chunk) ON (n.zone, n.search_priority);

CREATE INDEX concept_canonical_name IF NOT EXISTS
FOR (n:Concept) ON (n.canonical_name);

CREATE FULLTEXT INDEX concept_fulltext IF NOT EXISTS
FOR (n:Concept) ON EACH [n.canonical_name, n.alias_search_text, n.definition]
OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk', `fulltext.eventually_consistent`: true}};

CREATE INDEX semantic_review_status IF NOT EXISTS
FOR (n:Rule) ON (n.review_status);

CREATE INDEX refers_to_review_status IF NOT EXISTS
FOR ()-[r:REFERS_TO]-() ON (r.review_status);

// Sparse retrieval. Keep both original and contextualized text searchable.
CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
FOR (n:Chunk) ON EACH [n.text, n.contextualized_text, n.citation_label]
OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk', `fulltext.eventually_consistent`: true}};

// Dense retrieval. The embedding property must be a 3072-element LIST<FLOAT>.
CREATE VECTOR INDEX chunk_embedding_vector IF NOT EXISTS
FOR (n:Chunk) ON (n.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};
