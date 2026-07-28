// Rebuild the existing Chunk full-text index with Neo4j's CJK analyzer.
// Run once when upgrading a database created with standard-no-stop-words.
DROP INDEX chunk_fulltext IF EXISTS;

CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
FOR (n:Chunk) ON EACH [n.text, n.contextualized_text, n.citation_label]
OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk', `fulltext.eventually_consistent`: true}};
