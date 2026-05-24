# With real scraped data
python -m embeddings.ingest --file data/processed/catalog_latest.json

# Force re-embed everything
python -m embeddings.ingest --force